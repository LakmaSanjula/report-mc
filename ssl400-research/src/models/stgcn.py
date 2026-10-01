"""Spatial-Temporal Graph Convolutional Network for SSL400 Pose landmarks.

Input tensor shape: (N, C, T, V)
  N = batch
  C = channels (2 or 3)
  T = frames
  V = joints (33)

MC Dropout: nn.Dropout after global pool, before the classifier.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.pose_graph import Graph


def conv_init(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.BatchNorm2d | nn.BatchNorm1d, scale: float) -> None:
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution with learnable edge importance."""

    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, bias: bool = True) -> None:
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.num_subset = A.shape[0]
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subset, kernel_size=1, bias=bias)
        self.edge_importance = nn.Parameter(torch.ones(self.num_subset, A.shape[1], A.shape[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, _, t, v = x.size()
        x = self.conv(x)  # (N, out*K, T, V)
        x = x.view(n, self.num_subset, -1, t, v)
        y = 0
        A = self.A * self.edge_importance
        for i in range(self.num_subset):
            # (N, Cout, T, V) @ (V, V) over joint dimension
            y = y + torch.einsum("nctv,vw->nctw", x[:, i], A[i])
        return y


class STGCNBlock(nn.Module):
    """One spatial graph conv + temporal conv residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: np.ndarray,
        stride: int = 1,
        residual: bool = True,
        temporal_kernel: int = 9,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (temporal_kernel - 1) // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True) if dropout > 0 else nn.Identity(),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCN(nn.Module):
    """ST-GCN classifier with MC-Dropout-ready classifier dropout."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        num_joints: int = 33,
        channels: Optional[list[int]] = None,
        temporal_kernel: int = 9,
        dropout: float = 0.5,
        graph_strategy: str = "spatial",
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 64, 128, 256]
        if num_joints != 33:
            raise ValueError("This implementation expects MediaPipe Pose with 33 joints.")

        graph = Graph(strategy=graph_strategy)
        A = graph.A
        if A.shape[-1] != num_joints:
            raise ValueError(f"Graph size {A.shape[-1]} != num_joints {num_joints}")

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        layers: list[nn.Module] = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            stride = 2 if i in (2, 3) else 1  # temporal downsample on deeper blocks
            residual = i != 0
            # Keep block-internal dropout light; main MC dropout is before classifier.
            layers.append(
                STGCNBlock(
                    c_in,
                    c_out,
                    A,
                    stride=stride,
                    residual=residual,
                    temporal_kernel=temporal_kernel,
                    dropout=0.0,
                )
            )
            c_in = c_out
        self.st_gcn_layers = nn.ModuleList(layers)

        # Primary MC Dropout placement (plan): after pool, before classifier.
        self.mc_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

        bn_init(self.data_bn, 1)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        x = self.data_bn(x)
        x = x.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()

        for layer in self.st_gcn_layers:
            x = layer(x)

        # Global average pool over time and joints → (N, C)
        x = x.mean(dim=[2, 3])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.mc_dropout(feat)
        return self.fc(feat)


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode but activate Dropout modules for MC inference.

    Usage after loading a checkpoint:
        model.eval()
        enable_mc_dropout(model)
        for _ in range(N):
            logits = model(x)
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
