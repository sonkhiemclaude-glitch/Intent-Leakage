# Intent Leakage

Code and evaluation data for passive operational-intent inference from encrypted controller-facing traffic in an agentic intent-based networking pipeline.

## Measured scope

The matched dataset contains **504 runs**: 252 deterministic MCP executions and 252 native Amazon Bedrock Nova Micro executions, three operation labels (I01 quarantine, I02 release, I15 selected-rule audit), target counts 1/2/4, and two small ONOS/Mininet topologies. Each cohort has 14 sessions. Input bundles also retain unmatched collection rows; analyses select the common sessions rather than treating all input rows as matched.

The release includes **3,276 archived RF/kNN/CNN fits**, **216 offline feature-diagnostic fits**, a separate **eight-read loopback tunnel cohort**, and a **1,010-fit post-hoc extension** over the same 504 runs. The extension adds session-complete LOSO predictions, training/observation budgets, fixed classifier families, and blocked-label null checks. It adds no collection sessions, model-service requests, or confirmatory evidence. Split repeats overlap and are sensitivity checks, not independent confidence intervals. The code also contains a broader synthetic catalogue and shaping prototypes; their presence does not establish a measured fifteen-class benchmark or live defense efficacy.

Random-forest mean assurance class balanced accuracy is 90.83% for MCP and 100% for native execution (balanced chance 33.33%). Combined tool/assurance scope balanced accuracy is 91.11% and 95.56%. These are exploratory results on fixed implementations, with exact-intent binding for native execution. There is no measured all-channel privacy guarantee, unrestricted paraphrase generalization, second-model comparison, or paired live defense privacy/cost frontier.

## Verify the released results

Python 3.11+ is sufficient for integrity and prediction-metric verification; no AWS account, model call, model deserialization, or network connection is required:

```bash
python3 scripts/verify_results.py
```

The verifier checks SHA-256, archived input/model/prediction links, session separation, every stored prediction metric, all 1,010 extension predictions, the 252 saved LOSO model hashes, and engineering data counts. It does not deserialize models or fit classifiers. It writes a fresh summary under the ignored `results/` directory. Archived outputs retain their exploratory status.

## Reproduce the post-hoc extension

The frozen predictions, reports, protocol, and LOSO estimators are in `data/posthoc_extension_results.tar.gz`; the independently recomputed summary is `data/posthoc_extension_summary.json`. To refit the extension from the released classifier inputs, install the project dependencies, then run:

```bash
python scripts/extend_paper_evaluation.py \
  --archive data/analysis_results.tar.gz \
  --output results/posthoc_extension_refit \
  --cpus 2 --permutations 199 --store-oof-models
python scripts/summarize_extended_evaluation.py \
  --evaluation results/posthoc_extension_refit \
  --archive data/analysis_results.tar.gz \
  --engineering data/engineering_runs.json \
  --output results/posthoc_extension_summary.json --reload-models
python scripts/plot_publication_figures.py \
  --summary results/posthoc_extension_summary.json \
  --evaluation results/posthoc_extension_refit \
  --archive data/analysis_results.tar.gz \
  --ablation-archive data/ablation_results.tar.gz \
  --engineering data/engineering_runs.json \
  --engineering-captures data/readonly \
  --figure-dir results/posthoc_figures
```

The archived output preserves the exact published predictions. Refitting is exploratory and may not reproduce model bytes across library versions or hardware. The 95% ranges condition on the fixed LOSO fold models and resample the available sessions; they do not represent independent infrastructure replication.

## Aggregate revision supplement

`data/reviewer_summary.json` adds aggregate results for 12 count-only depth-two trees on one existing split and inference-only transfer of 12 saved forests to 36 separate archived direct-driver runs. It also reports class-level capacity statistics for 244 HTTP exchanges, four scheduler-condition summaries from eight local synthetic-handler cases (1,536 sender writes), successful-cohort accounting, and aggregate capture-replay outcomes. The destination executions were previously inspected and share the original infrastructure; they are not prospective validation. New execution was bounded smoke work, with no collection or cloud requests.

**This additional release contains only code and aggregate statistics.** Individual feature rows, predictions, run/session identifiers, trace sequences, per-exchange body observations, sender-write observations, endpoint manifests and additional PCAPs are not included. The existing release described above is separate from this supplement.

```bash
python3 scripts/verify_reviewer_summary.py
# Requires matplotlib and numpy; outputs stay under the ignored results directory.
python scripts/plot_reviewer_summary.py --output results/reviewer_summary_figures
```

The summary verifier checks SHA-256, recomputes balanced accuracy and macro-F1 from 24 confusion matrices, and checks count/quantile consistency. It does **not** independently reproduce private feature extraction, stored-model inference or raw scheduler quantiles. The plotter generates count-baseline, transfer and aggregate engineering plots. The body-capacity plot uses class maxima; private exchange-level ECDFs and individual trace timelines cannot be reconstructed from this supplement.

`scripts/reviewer_smoke.py` contains the bounded diagnostic procedure. It can regenerate detailed local smoke outputs from the previously released analysis and read-only inputs; generated files are not automatically published:

```bash
python scripts/reviewer_smoke.py --archive data/analysis_results.tar.gz \
  --readonly data/readonly --output results/reviewer_smoke --tshark tshark
```

`scripts/summarize_reviewer_results.py` documents the explicit field allowlist used to build the public summary. It requires the locally held detailed smoke and archived-transfer output folders, which are not supplied by this supplement. Aggregate arithmetic verification is therefore distinct from full raw-data reproduction. No manuscript or figure PDFs are included in this release.

## Install and test the implementation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,capture]'
python -m pytest -q -m 'not realtime'
```

Optional CNN and Bedrock dependencies: `pip install -e '.[cnn]'` and `pip install -e '.[aws]'`. Recorded fitting package versions are preserved in each archived report; a new fit with different libraries or hardware need not reproduce identical model bytes. The included lock file describes the development environment, not every archived fitting environment.

## Refit the attacks

Extract the analysis data into a **new empty directory**:

```bash
mkdir -p work/analysis
python3 -m tarfile -e data/analysis_results.tar.gz work/analysis
python scripts/evaluate_real_instrumentation.py --bundle work/analysis/outputs/rf_bundle/input.json --output results/refit_rf --cpus 4
python scripts/evaluate_real_sequences.py --bundle work/analysis/outputs/seq_bundle/input.json --output results/refit_knn --cpus 4 --max-length 512
# Requires the optional torch dependency:
python scripts/evaluate_real_cnn.py --bundle work/analysis/outputs/seq_bundle/input.json --output results/refit_cnn --cpus 4
python scripts/prepare_pre_ec2.py --archive data/analysis_results.tar.gz --output results/refit_diagnostics --cpus 4
```

The forest and neighbor splits hold sessions out. CNN additionally separates validation sessions and uses a smaller training budget; its scores are not an equal-budget ranking. Serialized model files are included for audit, but metric verification hashes them without loading pickle/joblib objects.

## Data layout

- `data/analysis_results.tar.gz`: classifier input bundles, frozen reports, predictions and fitted artifacts. Remote launch wrappers and old nested source bundles are excluded; research artifact bytes are unchanged.
- `data/ablation_results.tar.gz`: count/size/timing/full-feature and offline record-length-equalization diagnostics, including stored models and predictions.
- `data/posthoc_extension_results.tar.gz`: 1,010 stored post-hoc fits, predictions, protocol/checksum records, and LOSO model artifacts; all use the same matched 504-run corpus.
- `data/posthoc_extension_summary.json` and `data/posthoc_extension_model_reload.json`: independent metric recomputation and exact prediction-match receipt for reloaded LOSO models.
- `data/readonly/`: eight broad loopback PCAPs, selected-tunnel endpoint manifests and normalized visible TLS records. Use each flow manifest to exclude the unshaped backend connection.
- `data/engineering_runs.json`: body/IP accounting and 4,096 application-cell lateness observations from eight related transactions. The fixed active envelope is 2.56 seconds; this is not complete intent latency or a bandwidth-overhead baseline.
- `data/reviewer_summary.json`: aggregate-only revision diagnostics, confusion matrices, capacity/scheduler summaries and limitations.
- `data/provenance.json` and `data/SHA256SUMS.txt`: packaging provenance and dataset integrity checks.

Attack inputs include record-size/timing sequences and aggregate features. Raw attack-corpus PCAPs, machine-access records, infrastructure inventories and private operational logs are not included. Engineering PCAPs contain isolated emulated loopback traffic. The local original artifacts remain separate from this curated public dataset.

## Live collection boundary

`src/intent_leakage/` contains the adapters, orchestration, feature extraction, attackers, shaping and assurance code. Linux collection components are under `deployment/`; ONOS 2.7.0, Mininet 2.3.0, OVS 3.3.9 and TShark 4.2.2 were used in the measured testbed. Archived package versions and protocol settings describe the individual cohorts.

Live collection requires separately provisioned controller/network infrastructure and explicit external-execution configuration. Credentials are supplied locally and are never part of the dataset. The read-only tunnel does not implement remote mutation atomicity or cross-host fixed-cadence guarantees. Padding unused cells does not itself establish class indistinguishability.
