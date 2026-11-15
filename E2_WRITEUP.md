# E2: Rare-family thresholds (O2)

E2 measures how attack detection changes as training examples of Worms,
Shellcode, Backdoor, or Analysis are removed. `e2_rare_family.py` reduces one
family at a time, leaves all other families unchanged, and compares a full-data
baseline with 5%, 1%, 0.5%, and 0.1% retention. The default grid includes all four
shared classifiers, six implemented imbalance methods, and subsampling seeds
0, 1, and 2. Focal loss is excluded because the shared configuration does not
implement it.

## Sampling and tiny families

Counts are rounded to the nearest integer (half upward), with a minimum of one
record. Zero examples would be the E3 experiment instead. The code deliberately
does not use `MIN_RARE_RECORDS=6`: that would collapse all Worms levels into the
same experiment. Requested fraction, original count, and actual retained count
are saved in every row. Subsets are nested across levels within each seed and
are identical across classifiers and balancing methods.

With E3's deduplicated training data, the final-fit counts are:

| Family | Full | 5% | 1% | 0.5% | 0.1% |
|---|---:|---:|---:|---:|---:|
| Worms | 127 | 6 | 1 | 1 | 1 |
| Shellcode | 1,091 | 55 | 11 | 5 | 1 |
| Backdoor | 1,535 | 77 | 15 | 8 | 2 |
| Analysis | 1,594 | 80 | 16 | 8 | 2 |

Fold training counts are smaller. Fixed-neighbor SMOTE/ADASYN/SMOTE-ENN require
at least six examples of each class **within every training fold**; configurations
that cannot satisfy that requirement are recorded as `unsupported`. Other
estimator/resampler ValueErrors are recorded as `failed`, with the exception
message. Neither status is turned into zero recall or included in averages.
Adaptive neighbor counts would introduce a second experimental variable, so
the shared setting is preserved. MLP early stopping may also fail at tiny counts.

## Validation and scoring

Five stratified folds are generated from the unreduced official training
partition. Reduction happens only on each fold's training rows, so validation
retains its original family distribution. A family classifier is fitted, making
resampling act on family imbalance. Attack score is `1 - P(Normal)`.

The threshold maximizes attack F1 over pooled out-of-fold predictions; ties
choose the highest threshold. A fresh model is then fitted on the reduced full
training partition and evaluated on the unchanged test set at this threshold.
Test labels never choose thresholds or configurations. CV rows use the pooled
threshold and are diagnostic, not an independent estimate of threshold-selection
performance or inputs for an unbiased significance claim.

Shared `evaluate.py` supplies binary macro-F1, PR-AUC, average precision, ROC-AUC,
and false positives per 100,000 normal connections. `family_detection_recall`
means the tuned detector alerted on the family, regardless of predicted family.
`family_classification_recall` instead measures correct multiclass argmax labels;
it is independent of the tuned attack cutoff.

## Data modes and limitations

Default: `--raw` reads the configured official CSV paths, removes exact
within-partition duplicates, and fits preprocessing separately on each reduced
training fold. The final preprocessor is fitted only on reduced full training.
As in E3, train/test feature overlap is retained; see the preprocessing report
and E3 caveat when interpreting generalization.

Explicit exploratory mode: `--processed --processed-dir processed` reuses E3's processed data without requiring
its PR to merge. This is compatible with E3, but its encoder, medians, and protocol
grouping were fitted on the entire training partition. Consequently preprocessing
is not strictly isolated from validation, and it includes records later removed
by E2. This limitation is recorded in the experiment manifest. Scaling and
resampling still fit only on each reduced training fold. Use raw mode for the
strict validation protocol.

At the smallest levels, multiple requested fractions represent the same actual
sample count. They do not supply independent evidence of a finer detection
floor. The baseline is deterministic and repeated for paired comparisons; those
repeats do not measure model-seed uncertainty. Plot error bars are sample standard
deviations across subsampling seeds, not confidence intervals. Worms has only 44
test records in E3's data, so a single changed prediction changes recall by about
2.27 percentage points. A breakdown should be judged together with paired recall
changes and false alarms; the proposal does not specify a universal recall cutoff.
The generated `_floors.csv` makes an explicit operational choice: a mean
attack-detection recall decrease of at least 0.10 from the paired full baseline
is a breakdown (`--breakdown-drop` changes that criterion). It reports the first
tested fraction meeting that criterion as retention falls, the minimum actual
count tested, and the number of successful seeds. Missing configurations do not
establish successful detection below that floor. This brackets observed behavior
on the tested grid and does not establish a precise minimum sample requirement.

## Reproduction

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_e2 test_evaluate
# Exploratory grid using E3-compatible processed data:
.\.venv\Scripts\python.exe e2_rare_family.py --processed
# Strict fold-local preprocessing, with official CSVs in data/:
.\.venv\Scripts\python.exe e2_rare_family.py --raw --out results/e2_raw.csv
# Small real-data verification:
.\.venv\Scripts\python.exe e2_rare_family.py --processed --families Worms --classifiers decision_tree --methods none --levels 0.05 --seeds 0 --out results/e2_check.csv
```

Each completed configuration is appended to the results CSV; rerunning skips
saved configuration keys. CV rows are deduplicated during resume. A manifest
fingerprints data, shared configuration, and runner source to prevent mixing
incompatible experiments; use a new output path after changing them. Unsupported
and failed configurations are also skipped on resume; choose a new output path
to retry after resolving their cause.

Outputs next to the requested CSV are `_summary.csv` (mean, sample SD, count),
`_paired.csv` (seed-matched changes from the full baseline), `_unavailable.csv`,
`_floors.csv` (observed breakdowns under the explicit recall-loss criterion),
and one recall-versus-actual-count PNG per classifier. All experiment results
remain in the ignored `results/` directory. The full grid requires 1,440
configurations, up to 8,640 model fits before unsupported cases are excluded.

## Verification

The implementation passes 44 tests covering shared scoring, all four classifier
interfaces, all six implemented methods on synthetic data, family-preserving
subsampling, raw preprocessing with unseen categories, validation threshold
selection, unavailable configurations, CSV outputs, and resume deduplication.

A real-data check using E3's processed data ran the unbalanced Decision Tree on
Worms with seed 0, at full size and 5% retention (threshold 0.5 in both):

| Retention | Actual Worms training examples | Detection recall | Correct-family recall | False alarms per 100k normal |
|---|---:|---:|---:|---:|
| Full | 127 | 0.9545 | 0.5227 | 24,229.7 |
| 5% | 6 | 0.9545 | 0.0455 | 24,434.3 |

This illustrates why detecting an attack and naming its family are reported
separately. It is a pipeline check, not an E2 finding.

## Status and scope

- The official raw CSVs (175,341 training and 82,332 test rows, no missing
  values, no binary/family label inconsistencies) are expected at
  `data/UNSW_NB15_training-set.csv` and `data/UNSW_NB15_testing-set.csv`.
  The `data/` directory is gitignored.
- The runner and reporting pipeline are complete. The full 1,440-configuration
  raw-data grid (`results/e2_raw.csv`) is run separately; no aggregate result or
  final rare-family threshold is claimed here. Results will be added once the run
  finishes and `_coverage.csv` has been reviewed for failures.
- Focal loss is excluded. The proposal limits it to XGBoost and the shared
  configuration does not implement it.
- Fixed-k resampling cannot run on the tiniest samples; these cases are reported
  as `unsupported` rather than changing the experiment's sample sizes.
- CV metrics are threshold-selection diagnostics, not independent evidence for
  E6, which needs its own nested validation design.
- `--processed` mode reuses E3's processed CSVs and is exploratory only; it does
  not require E3's branch to be merged.
- `run_e2.py` wraps the runner with bounded native-library threading.
