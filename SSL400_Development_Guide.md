# Development Guide: Uncertainty-Aware Human-in-the-Loop Framework for Sinhala Sign Language Recognition

**Project basis:** Approved MCS 3204 proposal\
**Case study:** Sinhala Sign Language (SSL) using SSL400\
**Primary contribution:** uncertainty estimation, human verification,
and continual learning\
**Supporting prototype:** rule-based Sinhala sentence assembly and
LLM-assisted response

------------------------------------------------------------------------

## 1. Purpose and rules for using this guide

This document turns the proposal into an implementation plan, starting
from an empty development environment and progressing toward evaluation
and a working prototype.

### 1.1 What must be implemented

The proposal specifies:

1.  SSL400 dataset preparation.
2.  MediaPipe landmark-based representation.
3.  One recognition backbone: ST-GCN.
4.  Comparison of four uncertainty methods:
    -   Softmax confidence
    -   Predictive entropy
    -   Monte Carlo Dropout
    -   Temperature scaling
5.  An uncertainty-based accept/verify policy.
6.  Top-5 human verification.
7.  A replay memory containing only valid user-verified samples.
8.  Balanced replay and periodic continual-learning updates.
9.  Evaluation of recognition, calibration, inference time, forgetting,
    retained performance, feedback effort, robustness, usability, and
    the stated baselines/ablation.
10. A supporting rule-based Sinhala sentence-generation component and
    LLM demonstration.

### 1.2 Scope boundary

Do not add a second recognition backbone, another sign-language dataset,
another sign language, speech recognition, mobile deployment, user
accounts, or unrelated application features. They are not part of the
described project scope.

The proposal calls SSL400 a word-level dataset. Therefore, do not claim
that the system performs unrestricted continuous sentence translation.
The sentence assembly component is a supporting demonstration built from
recognized word labels, as described in the proposal.

### 1.3 Important implementation rule

Do not assume the exact CSV column names, landmark count, graph edges,
signer metadata, or split files. Inspect the downloaded dataset first.
The implementation must be based on the files actually supplied by
SSL400.

Where this guide says **decision to document**, record the choice,
reason, and evidence in the project report before proceeding.

------------------------------------------------------------------------

## 2. End-to-end system architecture

``` text
                  SSL400 dataset
            videos + landmark CSV files
                       |
                       v
             Dataset inspection/report
                       |
                       v
             Landmark data preparation
                       |
                       v
          Sequence tensor + class label
                       |
                       v
                    ST-GCN
                       |
                 class logits
                       |
                       v
           Uncertainty estimation methods
       Softmax | Entropy | MC Dropout | T scaling
                       |
                       v
          Calibrated accept/verify policy
                 /             \
          low uncertainty    high uncertainty
                |                  |
             accept             show Top-5
                                   |
                              user response
                                   |
                         valid verified sample?
                              /          \
                            yes           no
                             |             |
                        replay buffer   exclude
                             |
                   balanced replay sampling
                             |
                 periodic continual learning
                             |
                    evaluate updated model
                             |
                  retain/replace model by
                    documented criterion

Supporting demonstration, after recognition works:
recognized word sequence -> pause/buffer -> rule-based Sinhala
sentence assembly -> LLM prompt -> short Sinhala response
```

The research pipeline and supporting language demonstration should be
developed separately. A failure in the LLM component must not prevent
evaluation of the core recognition framework.

------------------------------------------------------------------------

## 3. Recommended development environment

### 3.1 Work split

Use the laptop for: - code editing and version control; - dataset
inspection and small preprocessing checks; - CSV validation; - webcam
and application development; - evaluation scripts and plots; - CPU
inference tests.

Use Google Colab when a GPU runtime is available for: - ST-GCN
training; - repeated uncertainty experiments; - continual-learning
experiments that are too slow locally.

Colab free GPU availability and usage limits can change. Do not assume a
GPU will always be available. Save checkpoints and experiment outputs to
persistent storage frequently. Keep the code runnable on CPU for
debugging, even if full training is performed in Colab.

### 3.2 Technology choices

  -----------------------------------------------------------------------
  Area                    Recommended choice      Notes
  ----------------------- ----------------------- -----------------------
  Language                Python 3.11 or 3.12     Choose a version
                                                  supported by the exact
                                                  PyTorch and MediaPipe
                                                  releases you install.
                                                  Pin it for
                                                  reproducibility.

  Deep learning           PyTorch                 Use one framework for
                                                  ST-GCN and uncertainty
                                                  experiments.

  Landmark extraction     MediaPipe               Use the dataset's
                                                  provided landmarks for
                                                  primary training if
                                                  they are valid and
                                                  match the proposal.

  Data handling           pandas, NumPy           Inspect and prepare CSV
                                                  data.

  Evaluation              scikit-learn            Classification metrics
                                                  and supporting
                                                  utilities.

  Video/webcam            OpenCV                  Capture frames for the
                                                  prototype.

  Plotting                Matplotlib              Confusion matrix,
                                                  reliability diagram,
                                                  and experiment plots.

  Development             VS Code                 Local development.

  Version control         Git                     Track code, not the
                                                  full dataset or large
                                                  checkpoints.

  Training                Google Colab Free, when GPU availability is not
                          available               guaranteed.

  Storage                 Google Drive or local   Store checkpoints and
                          SSD                     experiment outputs;
                                                  keep backups.
  -----------------------------------------------------------------------

**Version policy:** Do not copy version numbers from an old tutorial
without checking compatibility. On the day you create the environment,
install mutually compatible stable releases, record exact versions, and
freeze them. Do not upgrade packages during an experiment series unless
you create a new documented environment.

### 3.3 Create the project folder

Suggested structure:

``` text
ssl400-research/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/                 # downloaded data; do not commit
│   ├── inspected/           # reports about source files
│   └── processed/           # prepared arrays/tensors; do not commit
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   └── 02_model_debugging.ipynb
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── graph/
│   ├── models/
│   ├── uncertainty/
│   ├── policy/
│   ├── feedback/
│   ├── continual_learning/
│   ├── evaluation/
│   └── application/
├── tests/
├── checkpoints/             # do not commit large files
├── experiments/
└── results/
```

Create the folder, then initialize Git:

``` bash
mkdir ssl400-research
cd ssl400-research
git init
```

Create a Python virtual environment (example for Linux/WSL):

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activation is:

``` powershell
.venv\Scripts\Activate.ps1
```

Install packages in small groups after checking their compatibility.
Save exact installed versions:

``` bash
python --version
python -m pip freeze > requirements.txt
```

For Colab, create a notebook that prints the Python, PyTorch, CUDA, and
MediaPipe versions. The local and Colab environments need not be
identical in hardware, but the data format and model code must be
consistent.

### 3.4 `.gitignore`

At minimum, exclude:

``` gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/raw/
data/processed/
checkpoints/
*.pt
*.pth
*.ckpt
.env
```

Do not commit private credentials, API keys, full videos, or large
derived datasets.

------------------------------------------------------------------------

## 4. Phase 1 --- Acquire and inventory SSL400

### Objective

Obtain the dataset and establish exactly what is available before
writing preprocessing code.

### Steps

1.  Download SSL400 from the Kaggle location identified in the proposal.
2.  Keep the original downloaded archive unchanged.
3.  Extract it under `data/raw/`.
4.  List folders and files.
5.  Identify the original videos, preprocessed videos, CSV files,
    labels, and any train/validation/test split information.
6.  Read the dataset documentation and record its stated recording rate,
    duration, classes, and split.
7.  Do not rename or move individual source files until their naming
    pattern is understood.

### Create an inventory

Record:

  Item                            What to record
  ------------------------------- ---------------------------------------------
  Dataset version/download date   Exact source and date
  Class count                     Verify against files/labels
  Sample count                    Total and per split
  Video format                    Extension, dimensions, frame rate, duration
  CSV format                      File count, columns, row count
  Label mapping                   Class ID and class name
  Split definition                Supplied split or a split you create
  Signer metadata                 Whether signer IDs are available
  Missing/corrupt files           Counts and examples

### Completion check

Do not proceed until you can answer: - Where are the labels? - How is
each CSV linked to a video/sample? - What does one CSV row represent? -
How are frames ordered? - Which split does each sample belong to? - Is
signer identity available?

**Deliverable:** `results/dataset_inventory.md` plus a machine-readable
inventory CSV/JSON.

------------------------------------------------------------------------

## 5. Phase 2 --- Choose and verify the training representation

### Recommended primary input

Use the supplied MediaPipe landmark CSV as the primary training
representation **if inspection confirms that it contains the required
ordered landmark sequences and labels**. This is consistent with the
proposal's landmark-based ST-GCN approach and is more practical than
repeatedly extracting landmarks from raw videos during every training
run.

Retain the raw videos. They are useful for checking samples,
implementing webcam inference, and conducting the real-world-condition
evaluation.

### Do not assume the CSV schema

Open several CSVs, including samples from different classes and splits.
Inspect: - column names and data types; - frame index or timestamp; -
sample identifier and label; - landmark names or index convention; -
x/y/z or other coordinate fields; - confidence/visibility fields, if
present; - missing values; - number of frames per sample.

If the CSV files are combined into one table, determine how sample
boundaries are represented. If each sample has a separate CSV, verify
how the filename maps to the class label.

### Required decision record

Write `results/data_representation_decision.md` and state: 1. which
supplied representation is used for training; 2. why it matches the
proposal; 3. what fields are used; 4. how a sample is identified; 5. how
frame order is established; 6. what information is absent or uncertain.

**Stop condition:** If the CSV does not contain the expected ordered
landmarks, do not invent a tensor layout. Inspect the provided
preprocessed videos/raw videos and determine the supported extraction
route first.

------------------------------------------------------------------------

## 6. Phase 3 --- Dataset quality checks

Build a repeatable script, for example:

``` text
src/data/inspect_dataset.py
```

The script should report:

1.  number of samples per split;
2.  number of samples per class;
3.  minimum, maximum, and distribution of frames per sample;
4.  missing or duplicated sample identifiers;
5.  missing labels;
6.  malformed rows;
7.  missing/non-finite coordinate values;
8.  unexpected coordinate ranges;
9.  whether class IDs map consistently to class names;
10. whether samples overlap between splits.

Save the report instead of relying only on notebook output.

### Split leakage

Use the supplied split if it is documented and usable. Check whether the
same video/sample appears in more than one split. If signer IDs exist,
also inspect signer overlap.

Do not claim an unseen-signer evaluation unless the dataset provides
signer identity or you collect a separate appropriately labelled test
set. If signer IDs are unavailable, report this limitation and describe
exactly what robustness data you can legitimately evaluate.

### Completion check

The inspection script should run twice and produce the same counts for
an unchanged dataset.

**Deliverable:** dataset quality report and validation script.

------------------------------------------------------------------------

## 7. Phase 4 --- Build preprocessing

### Objective

Convert each sample into a consistent sequence representation suitable
for the selected ST-GCN implementation.

### Pipeline

``` text
source CSV
   -> parse sample and label
   -> order frames
   -> select documented landmark fields
   -> validate values
   -> handle missing values using a documented rule
   -> normalize coordinates using a documented rule
   -> form a fixed/consistent temporal sequence
   -> save sequence and label
```

### Decisions that must be made from the actual data

-   **Landmark set:** use only landmarks that exist in the supplied data
    and are justified by the proposal.
-   **Missing values:** determine whether missing landmarks mean
    detection failure, an unused landmark, or malformed data. Do not
    replace all missing values blindly with zero.
-   **Temporal length:** the proposal describes 3-second videos at 20
    FPS, which suggests 60 frames when the recording is complete. Verify
    actual CSV frame counts before choosing a sequence length.
-   **Sequence length mismatch:** document whether you use the complete
    sequence, temporal sampling, or padding/truncation. Apply the same
    rule to training and inference.
-   **Normalization:** define the reference point and scale from the
    available body/hand landmarks. Fit no data-dependent normalization
    statistics using the test set.
-   **Augmentation:** do not add transformations unless they are
    justified, label-preserving for SSL, and documented as part of the
    experimental method.

### Data split discipline

Fit any learned preprocessing parameters using training data only. Use
validation data for model selection and calibration. Keep the test split
untouched until final evaluation.

### Output format

Choose and document one consistent representation, for example a NumPy
array or PyTorch tensor with dimensions corresponding to:

``` text
time × landmarks × coordinates
```

This is a conceptual shape, not a claim about the actual SSL400 CSV.
Confirm the exact dimensions after inspection.

Save: - processed sequences; - integer labels; - class-to-index
mapping; - sample IDs; - split membership; - preprocessing
configuration/version.

### Completion checks

-   A sample can be loaded and visualized.
-   Its label matches the source.
-   Frame order is correct.
-   The same preprocessing code works for training and a single
    inference sample.
-   No test samples were used to fit preprocessing parameters.

**Deliverable:** reproducible preprocessing script and processed-data
manifest.

------------------------------------------------------------------------

## 8. Phase 5 --- Define the ST-GCN graph

### Objective

Represent the selected landmarks as graph nodes and define spatial and
temporal connections for ST-GCN.

### Steps

1.  Create a landmark index map from the actual CSV schema.
2.  Define which nodes belong to the hand, body/pose, and face groups,
    where those groups are present.
3.  Define spatial edges only where the landmark topology supports them.
4.  Define temporal connections between the same landmark in adjacent
    frames.
5.  Record node count, edge list, and coordinate channels.
6.  Test graph construction with one sample and visualize it if
    possible.
7.  Verify that the graph node order exactly matches the preprocessing
    output order.

Do not invent anatomical connections for landmarks whose identity is
unknown. The graph definition is a research implementation decision and
must be documented.

### ST-GCN input contract

Write a small test that asserts: - expected tensor dimensions; -
expected number of nodes; - valid label range; - finite input values; -
graph indices within the node range.

**Deliverable:** graph definition file and unit tests.

------------------------------------------------------------------------

## 9. Phase 6 --- Train the recognition-only ST-GCN baseline

### Objective

Establish a reliable recognition model before adding uncertainty or
continual learning.

### Steps

1.  Implement one ST-GCN backbone.
2.  Use the training split for gradient updates.
3.  Use the validation split for model selection and early stopping
    decisions.
4.  Save the best checkpoint according to a predeclared validation
    criterion.
5.  Record random seed, configuration, package versions, training time,
    and hardware.
6.  Run the final model once on the untouched test split.
7.  Save predictions and per-sample results.

### Training configuration to record

  Setting             Record
  ------------------- ----------------------------------
  Input dimensions    Actual verified tensor shape
  Number of classes   Verified class count
  Graph               Version/hash of graph definition
  Optimizer           Name and settings
  Learning rate       Exact value and schedule
  Batch size          Exact value
  Epochs              Maximum and actual
  Loss                Exact function
  Seed                Exact seed
  Checkpoint rule     Validation selection criterion
  Hardware            CPU/GPU model and runtime

Do not choose settings based on test-set performance.

### Required baseline metrics

-   accuracy;
-   precision;
-   recall;
-   F1-score;
-   inference time.

Report macro and weighted precision/recall/F1 where appropriate, and
explain the averaging method. Preserve the class-level results because
aggregate scores can hide minority-class problems.

**Deliverable:** trained ST-GCN checkpoint, configuration, logs, test
predictions, and baseline report.

------------------------------------------------------------------------

## 10. Phase 7 --- Implement uncertainty estimation

Use the same trained ST-GCN backbone for all four methods. Keep the test
set out of method selection and calibration.

### 10.1 Softmax confidence

1.  Obtain logits from ST-GCN.
2.  Apply softmax to obtain class probabilities.
3.  Use the maximum class probability as the confidence score.
4.  Store predicted class, confidence, and the complete probability
    vector.

### 10.2 Predictive entropy

1.  Use the predicted probability distribution.
2.  Calculate entropy consistently across samples.
3.  Define whether the stored value is entropy (higher means more
    uncertain) or a transformed confidence score.
4.  Keep this convention consistent in the policy and plots.

### 10.3 Monte Carlo Dropout

1.  Include dropout in the model in a documented way.
2.  At inference, enable dropout for repeated stochastic forward passes
    while keeping other model behavior controlled.
3.  Run a predeclared number of passes.
4.  aggregate the probability predictions using a documented method;
5.  calculate the chosen uncertainty statistic from the repeated
    predictions;
6.  record the additional inference time.

Do not silently change the number of passes between experiments.

### 10.4 Temperature scaling

1.  Collect logits on the validation split.
2.  Fit one scalar temperature using validation data only.
3.  Apply the temperature to logits before softmax.
4.  Evaluate calibration on validation data while selecting the method.
5.  Freeze the temperature before final test evaluation.

Temperature scaling is a post-hoc calibration step; it does not require
retraining the ST-GCN backbone.

### Common output contract

Each method should produce a consistent record:

``` text
sample_id
true_label
predicted_label
probabilities
uncertainty_or_confidence
inference_time
method_name
```

**Deliverable:** four tested uncertainty implementations and saved
validation/test outputs.

------------------------------------------------------------------------

## 11. Phase 8 --- Compare calibration and select a policy score

Compare the four methods using the proposal's measures:

-   Expected Calibration Error (ECE);
-   Brier score;
-   reliability diagram;
-   coverage versus risk;
-   inference time.

### Definitions to keep consistent

-   **Coverage:** proportion of samples automatically accepted.
-   **Selective risk:** error rate among automatically accepted samples.
-   **ECE:** report the binning method and number of bins.
-   **Brier score:** state whether it is multiclass and how it is
    averaged.
-   **Reliability diagram:** state how confidence bins are formed.

Use validation data to choose the uncertainty method and decision
threshold. Use the test set only for the final, locked evaluation.

### Threshold procedure

1.  Define the target automatic-acceptance risk before threshold
    selection. The proposal gives 5% as an example tolerance; confirm
    the target with the supervisor.
2.  Calculate validation coverage and risk over candidate thresholds.
3.  Select a threshold that meets the target on validation data, if one
    exists.
4.  Report the resulting validation coverage.
5.  If no threshold meets the target, report that fact; do not
    manufacture a passing threshold.
6.  Freeze the method and threshold.
7.  Evaluate the frozen policy on test data and report both coverage and
    risk.

A confidence threshold and an uncertainty threshold may run in opposite
directions. Make the direction explicit in code and documentation.

**Deliverable:** uncertainty comparison table, reliability plots,
coverage-risk plot, and frozen policy configuration.

------------------------------------------------------------------------

## 12. Phase 9 --- Implement accept/verify routing

### Required behavior

``` text
model prediction + selected uncertainty score
                    |
                    v
             frozen threshold
              /           \
       accepted region   uncertain region
             |                 |
        auto-accept       display Top-5
                               |
                         user response
```

### Top-5 verification behavior

For an uncertain prediction: 1. display the five highest-ranked
candidate classes and their labels; 2. allow the user to confirm the
first candidate; 3. allow the user to select a corrected label from the
displayed five; 4. record rejection or unresolved status; 5. mark a
sample as verified only when the user confirms the top-1 or selects a
corrected top-5 label.

Rejected or unresolved samples must not enter the replay memory.

The proposal only defines correction within the Top-5 list. Do not
silently add an unrestricted label-search feature. If the correct label
is not in the Top-5, record the sample as rejected/unresolved under the
proposal's policy.

### Store an audit record

For each verification event, record: - sample ID; - model version; -
predicted class; - displayed Top-5; - confidence/uncertainty; - user
action; - verified label, if any; - timestamp or session identifier, if
needed for analysis; - time taken to verify.

Avoid collecting personal information that is not needed for the
research.

**Deliverable:** tested routing logic and verification-event records.

------------------------------------------------------------------------

## 13. Phase 10 --- Build the replay memory

### Objective

Store only verified samples that are valid for retraining.

A replay record should contain: - stable sample ID; - processed landmark
sequence or a reference to it; - verified class label; - original
prediction; - uncertainty/confidence values; - model version; -
verification action; - source split or collection source.

Do not put rejected or unresolved samples into the training buffer.

### Storage choice

For an initial research prototype, a structured metadata file plus saved
array files is sufficient. A database is not required by the proposal.

### Integrity checks

-   verified label is in the known class mapping;
-   sample tensor passes the same preprocessing validation;
-   no duplicate sample is unintentionally counted repeatedly;
-   every buffer entry has a traceable verification record.

**Deliverable:** replay-buffer module and tests.

------------------------------------------------------------------------

## 14. Phase 11 --- Balanced replay and retraining trigger

### Trigger policy from the proposal

Retraining begins when either: - a predefined number of new verified
samples has accumulated; or - a minority class falls below a minimum
representation threshold in the buffer;

whichever occurs first.

The actual sample count and minimum representation value are not
specified in the proposal. Select them through a documented pilot and
supervisor agreement. Do not present them as proposal-provided values.

### Balanced sampling

1.  Group verified samples by class.
2.  Inspect class counts.
3.  Apply the documented balanced replay sampler.
4.  Record how many samples from each class were selected.
5.  Keep the sampling rule fixed for comparable experiments.

### Retraining procedure

1.  Load the current model checkpoint.
2.  Construct the incremental training batch from verified samples and
    replay samples according to the documented policy.
3.  Train using the same ST-GCN architecture.
4.  Evaluate on the fixed validation set.
5.  Evaluate retained performance on the fixed original test set at
    scheduled rounds.
6.  Save the candidate checkpoint and full configuration.
7.  Apply a predeclared model-update/retention rule.

The proposal requires periodic model updates and forgetting analysis but
does not fully specify a checkpoint promotion rule. Define this with the
supervisor before experiments. Do not replace the deployed model solely
because training loss decreased.

**Deliverable:** trigger implementation, balanced sampler, retraining
script, and round-by-round logs.

------------------------------------------------------------------------

## 15. Phase 12 --- Evaluate continual learning and forgetting

At each retraining round, record:

  -----------------------------------------------------------------------
  Measure                             Meaning
  ----------------------------------- -----------------------------------
  Performance before retraining       Baseline immediately before the
                                      update

  Performance after retraining        Performance after the update

  Original test performance           Retained performance on the fixed
                                      test split

  Per-class performance               Detect classes that degrade

  Forgetting                          Accuracy decrease on previously
                                      learned classes

  Verified samples used               Number and class distribution

  Feedback effort                     Verification actions and time

  Retraining cost                     Training duration and resources
  -----------------------------------------------------------------------

Use the same fixed evaluation set across rounds so changes are
comparable. Do not add newly verified samples to the fixed test set.

### Feedback effort

Measure: - number of verification actions per session; - average time to
verify an uncertain prediction.

Define exactly how timing starts and ends, and use the same procedure
across conditions.

**Deliverable:** continual-learning results table and plots over rounds.

------------------------------------------------------------------------

## 16. Phase 13 --- Implement the required baselines and ablation

Keep the ST-GCN backbone and data split consistent across comparisons.

### Required comparisons

  -----------------------------------------------------------------------
  Experiment                          Configuration
  ----------------------------------- -----------------------------------
  Recognition-only baseline           ST-GCN without uncertainty routing
                                      or retraining

  Uncertainty without continual       Uncertainty and verification, but
  learning                            verified samples do not update the
                                      model

  Random feedback selection           Same feedback volume, but sample
                                      selection is random

  Full framework                      Uncertainty, verification, replay,
                                      balanced sampling, and continual
                                      learning
  -----------------------------------------------------------------------

### Ablation sequence

1.  ST-GCN only.
2.  ST-GCN + uncertainty.
3.  ST-GCN + uncertainty + human verification.
4.  Full framework with continual learning.
5.  Random-feedback-selection comparison.

For a fair comparison: - use the same backbone; - use the same initial
split; - control the number of feedback samples; - use comparable
training budgets; - record random seeds; - report variation across
repeated runs if feasible.

Do not describe one method as better unless the measured results support
that conclusion.

**Deliverable:** reproducible experiment configurations and comparison
report.

------------------------------------------------------------------------

## 17. Phase 14 --- Real-world robustness evaluation

The proposal names: - unseen signers; - lighting conditions; - camera
angles; - backgrounds; - signing speeds.

First determine what the existing dataset can test. If it does not
contain metadata or samples for a condition, a separate controlled
recording/evaluation set may be needed. This is not permission to add
another public dataset; it is a way to evaluate the conditions already
stated in the proposal.

### Protocol

1.  Define each condition before recording/testing.
2.  Keep the class labels and recording procedure documented.
3.  Separate robustness evaluation samples from training and calibration
    data.
4.  Do not use robustness test samples to tune the threshold.
5.  Report recognition performance, uncertainty, false automatic
    acceptance, verification rate, and inference time by condition.
6.  Document participant consent and university ethics requirements
    before collecting human video.

If unseen-signer data cannot be obtained, state that limitation clearly
rather than claiming unseen-signer validation.

**Deliverable:** robustness protocol, condition-wise results, and
limitations.

------------------------------------------------------------------------

## 18. Phase 15 --- Real-time webcam prototype

Build this only after the offline model and preprocessing pipeline are
stable.

### Runtime flow

``` text
webcam frame
   -> frame handling
   -> MediaPipe landmarks
   -> temporal sequence buffer
   -> same preprocessing used in training
   -> ST-GCN
   -> selected uncertainty method
   -> accept/verify policy
   -> accepted label or Top-5 verification
```

### Critical consistency requirement

The webcam path must produce the same landmark order, coordinate
convention, normalization, and temporal dimensions used during training.
If they differ, predictions are not a valid test of the trained model.

### Build in small tests

1.  Open webcam and display frames.
2.  Extract landmarks from one frame.
3.  Confirm landmark order and values.
4.  Accumulate a sequence.
5.  Confirm sequence shape.
6.  Run one saved model prediction.
7.  Run uncertainty estimation.
8.  Display accepted label or Top-5.
9.  Measure end-to-end latency.

Do not describe the system as real-time until measured end-to-end timing
supports that claim.

**Deliverable:** local webcam demonstration and measured inference time.

------------------------------------------------------------------------

## 19. Phase 16 --- Supporting rule-based Sinhala sentence assembly

This is not the core research contribution. Implement only the four
stages named in the proposal:

1.  Tokenization.
2.  Role assignment.
3.  Sentence-type classification.
4.  Grammar building.

### Flow

``` text
recognized word labels
   -> buffer sequence
   -> detect signing pause
   -> tokenize
   -> assign roles
   -> classify sentence type
   -> build Sinhala SOV sentence
```

The proposal does not provide a complete Sinhala grammar rule set or a
labelled sentence dataset. Therefore: - define a small, explicit rule
set for the prototype; - document the supported patterns and
limitations; - do not claim broad Sinhala language understanding; - test
with examples reviewed by a competent Sinhala speaker.

The sentence assembler should consume accepted/verified recognition
outputs according to the documented design. It should not silently treat
uncertain predictions as confirmed words.

**Deliverable:** rule-based module, supported-pattern documentation, and
example tests.

------------------------------------------------------------------------

## 20. Phase 17 --- LLM demonstration

The proposal describes sending structured Sinhala text to an LLM using a
prompt template to produce a polite, short, Sinhala-only response.

Implement this only after sentence assembly works.

### Steps

1.  Choose the API named in the proposal or another permitted equivalent
    only if necessary.
2.  Check the provider's current free-tier terms and availability before
    relying on it.
3.  Keep the API key outside source control, for example in an
    environment variable.
4.  Send only the generated Sinhala text and the minimal instruction
    needed.
5.  Request a short, polite, Sinhala-only response.
6.  Display the response.
7.  Handle network/API errors without breaking the recognition system.
8.  Record that this is a supporting demonstration, not the evaluated
    core contribution.

A hosted API may change its free quota or terms. The core research must
remain demonstrable without the LLM service.

**Deliverable:** optional LLM-connected demonstration and documented
limitations.

------------------------------------------------------------------------

## 21. Phase 18 --- Usability and user-satisfaction evaluation

The proposal includes usability and user satisfaction, and also defines
feedback effort measures.

Before involving participants: 1. prepare a short, consistent task
procedure; 2. define what counts as a verification action; 3. define how
time-to-verify is measured; 4. prepare the questions or rating form; 5.
obtain required supervisor/university ethics approval; 6. explain the
study and obtain informed consent; 7. avoid collecting unnecessary
personal data.

Report participant count, task procedure, missing responses, and
limitations. Do not claim that a small convenience sample represents all
SSL users.

**Deliverable:** approved evaluation protocol and anonymized results.

------------------------------------------------------------------------

## 22. Phase 19 --- Final evaluation and reporting

Organize the results under the proposal's evaluation categories.

### A. Recognition

-   accuracy;
-   precision;
-   recall;
-   F1;
-   inference time.

### B. Uncertainty/calibration

-   ECE;
-   Brier score;
-   reliability diagrams;
-   coverage versus risk;
-   inference overhead.

### C. Continual learning

-   before/after performance;
-   forgetting;
-   retained performance over rounds;
-   class-wise effects.

### D. Human feedback

-   verification actions per session;
-   average time-to-verify;
-   verified sample count and class distribution.

### E. Robustness

-   unseen signer, where supported;
-   lighting;
-   background;
-   camera angle;
-   signing speed.

### F. System demonstration

-   webcam pipeline;
-   Top-5 verification;
-   rule-based Sinhala sentence output;
-   LLM response demonstration.

### G. Baselines and ablation

Report the required configurations using consistent metrics and
experimental conditions.

Every result should be traceable to: - a code version/commit; - a
configuration; - a dataset split; - a model checkpoint; - a random
seed; - an output file.

------------------------------------------------------------------------

## 23. Suggested implementation milestones

  -----------------------------------------------------------------------
  Milestone               Work                    Exit condition
  ----------------------- ----------------------- -----------------------
  M1                      Environment and         Environment works;
                          repository              versions recorded

  M2                      Dataset inventory       Files, labels, splits,
                                                  and metadata understood

  M3                      CSV validation          Repeatable quality
                                                  report produced

  M4                      Preprocessing           Valid sequences and
                                                  manifest generated

  M5                      Graph definition        Node order/edges
                                                  documented and tested

  M6                      ST-GCN baseline         Checkpoint and baseline
                                                  metrics saved

  M7                      Uncertainty methods     Four methods return
                                                  consistent outputs

  M8                      Calibration and policy  Validation-selected
                                                  method/threshold frozen

  M9                      Verification            Top-5 actions and audit
                                                  records work

  M10                     Replay and continual    Verified-only buffer
                          learning                and periodic update
                                                  work

  M11                     Research comparisons    Baselines and ablation
                                                  completed

  M12                     Robustness              Condition-wise protocol
                                                  and results completed

  M13                     Webcam prototype        End-to-end pipeline
                                                  demonstrated and timed

  M14                     Supporting language     Rule-based output
                          layer                   tested

  M15                     LLM demonstration       Optional supporting
                                                  response works

  M16                     Final evaluation        Results and limitations
                                                  documented
  -----------------------------------------------------------------------

Do not move to the next milestone merely because code runs. Meet the
exit condition.

------------------------------------------------------------------------

## 24. Experiment and file naming

Use consistent names, for example:

``` text
experiments/
  baseline_stgcn_seed01.yaml
  uncertainty_softmax.yaml
  uncertainty_entropy.yaml
  uncertainty_mc_dropout.yaml
  uncertainty_temperature_scaling.yaml
  continual_learning_full.yaml
  continual_learning_random_feedback.yaml

results/
  dataset_inventory.md
  dataset_quality.csv
  baseline_metrics.csv
  calibration_metrics.csv
  coverage_risk.csv
  continual_learning_rounds.csv
  robustness_results.csv
  feedback_effort.csv
```

For each run, save: - configuration; - software versions; - seed; -
start/end time; - hardware; - training log; - checkpoint path; -
validation metrics; - test metrics, only for the final locked
evaluation.

------------------------------------------------------------------------

## 25. Common mistakes to avoid

1.  Training before understanding the CSV structure.
2.  Assuming every sample has exactly 60 usable frames without checking.
3.  Mixing samples from the same video across splits.
4.  Claiming unseen-signer testing without signer information or a
    separate test protocol.
5.  Selecting the uncertainty threshold using test data.
6.  Using rejected/unresolved samples for retraining.
7.  Retraining after every correction despite the proposal's periodic
    trigger.
8.  Comparing methods with different feedback volumes or inconsistent
    test sets.
9.  Changing preprocessing between training and webcam inference.
10. Reporting only overall accuracy and hiding class-wise performance.
11. Claiming unrestricted sentence translation from a word-level
    dataset.
12. Spending substantial time on the LLM before the core research
    pipeline works.
13. Depending on a free GPU/API being available at all times.
14. Saving checkpoints without their configuration and software
    versions.

------------------------------------------------------------------------

## 26. First practical task: what to do now

Start with only these tasks:

-   [ ] Create the `ssl400-research` folder and Git repository.
-   [ ] Create and activate the Python environment.
-   [ ] Record Python and package versions.
-   [ ] Download SSL400 without modifying the source archive.
-   [ ] Extract it under `data/raw/`.
-   [ ] List the directory structure.
-   [ ] Open several CSV files and record their columns.
-   [ ] Identify the label and sample-ID mapping.
-   [ ] Verify frame ordering and sequence lengths.
-   [ ] Confirm the supplied train/validation/test split.
-   [ ] Check whether signer IDs are available.
-   [ ] Write the dataset inventory report.

**Do not start ST-GCN training until these checks are complete.**

The next implementation step after this checklist is to write the
dataset inspection script using the actual SSL400 files. The CSV schema
must be known before writing a correct loader or defining the ST-GCN
input dimensions.

------------------------------------------------------------------------

## 27. Proposal traceability

  Proposal requirement                   Where implemented in this guide
  -------------------------------------- ---------------------------------
  SSL400 dataset                         Phases 1--4
  MediaPipe landmarks                    Phases 2--4 and 15
  Single ST-GCN backbone                 Phases 5--6
  Four uncertainty methods               Phase 7
  Calibration and coverage-risk          Phase 8
  Accept/verify and Top-5                Phase 9
  Verified-only replay memory            Phase 10
  Balanced replay and periodic trigger   Phase 11
  Forgetting and retained performance    Phase 12
  Required baselines and ablation        Phase 13
  Real-world conditions                  Phase 14
  Real-time recognition                  Phase 15
  Rule-based Sinhala sentence assembly   Phase 16
  LLM supporting prototype               Phase 17
  Usability/user satisfaction            Phase 18
  Final metrics and report               Phase 19

This guide is an implementation plan, not evidence that any model has
already been trained or that any performance target has been achieved.
All thresholds, graph details, preprocessing rules, and training
settings that the proposal leaves open must be selected, justified, and
recorded during implementation.
