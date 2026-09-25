# MLOps ML Pipeline

## 1. Project Overview

This project implements a small machine-learning pipeline for binary classification. The workflow validates the committed dataset, trains a baseline and a candidate model, checks the candidate with an F1-score quality gate, runs application tests, and publishes a model package only when the preceding steps pass.

```text
Dataset -> Validation -> Baseline -> Candidate Model -> Quality Gate -> Application Tests -> Model Package
```

The implementation is written as Python scripts and is executed by GitHub Actions in one job.

## 2. Dataset and Task

| Item | Actual project detail |
|---|---|
| Dataset name | Breast Cancer Wisconsin (Diagnostic) Data Set |
| Dataset source | Kaggle dataset `uciml/breast-cancer-wisconsin-data`, downloaded in the accompanying notebook with `kagglehub` |
| File location | `data/breast_cancer_wisconsin.csv` |
| Rows | 569 |
| Input features | 30 |
| Target | `diagnosis` |
| Task | Binary classification; the CSV target values are `0` and `1` |
| Download behavior | The notebook downloads the source dataset, but the pipeline uses the prepared CSV committed in the repository; the GitHub Actions workflow has no download step |

The 30 input features are:

```text
radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
area_worst, smoothness_worst, compactness_worst, concavity_worst,
concave_points_worst, symmetry_worst, fractal_dimension_worst
```

The notebook reads the source `data.csv`, removes `id` and `Unnamed: 32`, and converts diagnosis labels from `M`/`B` to `1`/`0`. The resulting CSV is a committed, labeled dataset with a target column and a fixed set of numeric input columns for a classification pipeline.

## 3. Model and Baseline

`src/train.py` uses the following models:

| Role | Model and configuration |
|---|---|
| Baseline | `DummyClassifier(strategy="most_frequent")` |
| Candidate | `RandomForestClassifier(n_estimators=100, random_state=42)` |

The data is split with `train_test_split` using `test_size=0.20` and `random_state=42`. Stratification is not enabled because no `stratify` argument is passed. There is no preprocessing step in the training code. Both models are trained and evaluated using the same split and therefore the same validation/test data.

## 4. Evaluation Metric and Quality Gate

The quality-gate metric is F1-score. F1-score combines precision and recall, making it useful here because a classification result should account for both false positives and false negatives rather than relying only on accuracy.

The implemented rule is:

```text
model F1 >= baseline F1 + margin
```

The margin in `src/train.py` is `0.05`. If the candidate does not meet this required score, training exits with status 1 before saving the model and metrics. A margin that is too low would make it easier for a weak candidate to pass; a margin that is too high could reject a useful candidate. The workflow uses the value currently implemented in the script.

## 5. Data Validation

`src/validate.py` checks:

- The dataset file exists at `data/breast_cancer_wisconsin.csv`.
- All expected feature columns are present.
- The `diagnosis` target column is present.
- The dataset is not empty.
- The target column contains no missing values.

When a required validation check fails, the script calls `sys.exit(1)`, so the pipeline exits with a non-zero status. On success it prints the row count, feature count, and target name.

## 6. Application Tests

`tests/test_prediction.py` contains three tests:

1. The saved model can be loaded with `joblib`.
2. A sample containing all required features produces a prediction with shape `(1,)`.
3. A sample missing a required feature is rejected with a `ValueError` containing `Missing required feature`.

The successful workflow evidence reports `3 passed`.

## 7. GitHub Actions CI Pipeline

The workflow is [.github/workflows/ml-pipeline.yaml](.github/workflows/ml-pipeline.yaml). It runs on pushes to the `main` branch and executes these steps in the `ml-pipeline` job:

1. Checkout repository with `actions/checkout@v4`.
2. Set up Python 3.11 with `actions/setup-python@v5`.
3. Install dependencies from `requirements.txt`.
4. Validate the dataset with `python src/validate.py`.
5. Train and evaluate the quality gate with `python src/train.py`.
6. Run the application tests with `pytest`.
7. Prepare the model package by copying the generated model, metrics, prediction script, and requirements file.
8. Upload the package with `actions/upload-artifact@v4`.

Training and tests run in the same GitHub Actions job. Because the later steps depend on earlier steps succeeding, a failed validation, quality gate, or test prevents package preparation and artifact upload.

## 8. Model Package / Artifact

The package is published as a GitHub Actions artifact, not committed to the repository. The package contains:

- `model.joblib` - trained Random Forest model
- `metrics.json` - baseline, candidate, metric, split, and quality-gate information
- `predict.py` - prediction function and required feature definition
- `requirements.txt` - runtime dependencies

The latest successful run currently produced `model-package-8`. The artifact name is built from the workflow run number:

```text
model-package-${{ github.run_number }}
```

## 9. Failure and Recovery Demonstration

### Failure A: Model quality failure

A deliberate model-quality failure is not demonstrated by the available repository history or GitHub Actions evidence. No available failed run shows the quality-gate step failing because `model F1 < baseline F1 + 0.05`. This section still needs a dedicated quality-failure run before it can be reported as completed.

### Failure B: Application test failure

| Item | Evidence |
|---|---|
| Run | [Run 5](https://github.com/RiteshRG/mlops-ml-pipeline/actions/runs/36108504983) |
| Run number | 5 |
| Commit | `b88b367` |
| Change | The test file was changed to add repository-path setup/imports, while the prediction feature names were still incompatible with the trained model feature names. |
| Why it failed | The workflow's `Run application tests` step failed. |
| Exact check that prevented publication | The `pytest` application-test step failed. |
| Artifact upload | Skipped; `Prepare model package` and `Upload model package` were skipped. |
| Recovery | The prediction feature list was corrected in commit `3dddae0`; run 7 then completed successfully and uploaded `model-package-7`. |

The public run evidence proves the failed workflow step and skipped publication. The exact pytest traceback is not exposed on the public page without sign-in.

## 10. Successful Run

The latest successful workflow is [run 8](https://github.com/RiteshRG/mlops-ml-pipeline/actions/runs/36109637276), triggered by commit `cffbabd`.

| Result | Value |
|---|---:|
| Baseline F1 | 0.0000 |
| Random Forest F1 | 0.9524 |
| Margin | 0.0500 |
| Required F1 | 0.0500 |
| Quality gate | PASSED |
| Application tests | 3 passed |
| Artifact | `model-package-8` |

## 11. CI and Artifact Delivery

Continuous integration is demonstrated by the GitHub Actions workflow automatically running validation, training, quality-gate evaluation, and `pytest` on every push to `main` in a clean Ubuntu/Python 3.11 job. Artifact delivery is demonstrated by packaging the generated files and uploading them with `actions/upload-artifact@v4` after all required checks pass.

## 12. MLOps Maturity

This is an early, repeatable CI-focused MLOps implementation. It has committed data, scripted validation/training/prediction, a baseline comparison, a quality gate, automated tests, and CI artifact publication.

It does not implement production deployment, a model registry, monitoring, drift detection, automated retraining, or a deployed prediction service. Those capabilities would be needed to move beyond this basic CI and artifact-delivery stage.

## 13. Reproducibility

Reproducibility is supported by:

- `random_state=42` in the train/test split and Random Forest.
- A fixed `test_size=0.20` split configuration.
- The committed dataset at `data/breast_cancer_wisconsin.csv`.
- Dependencies listed in `requirements.txt`.
- Python scripts rather than notebooks.
- A GitHub Actions environment using Python 3.11.

## 14. Repository Structure

```text
mlops-ml-pipeline/
|-- .github/
|   `-- workflows/
|       `-- ml-pipeline.yaml
|-- data/
|   `-- breast_cancer_wisconsin.csv
|-- Breast_Cancer_Wisconsin_(Diagnostic)_Data_Set.ipynb
|-- src/
|   |-- __init__.py
|   |-- predict.py
|   |-- train.py
|   `-- validate.py
|-- tests/
|   `-- test_prediction.py
|-- requirements.txt
`-- README.md
```

The `artifacts/` and `model-package/` contents are generated at runtime. The model package is uploaded by GitHub Actions and is not part of the committed repository tree.

## 15. How to Run Locally

From the repository root:

```powershell
python -m pip install -r requirements.txt
python src/validate.py
python src/train.py
pytest
```

`python src/train.py` writes `artifacts/model.joblib` and `artifacts/metrics.json` only after the quality gate passes. The tests expect the saved model to exist.

## 16. Submission / Evidence

- Failure A GitHub Actions run: **Not yet demonstrated; replace this placeholder after the deliberate quality-failure run is completed.**
- Failure B GitHub Actions run: [Run 5](https://github.com/RiteshRG/mlops-ml-pipeline/actions/runs/36108504983)
- Final successful GitHub Actions run: [Run 8](https://github.com/RiteshRG/mlops-ml-pipeline/actions/runs/36109637276)
- Successful downloadable artifact: [model-package-8 in run 8](https://github.com/RiteshRG/mlops-ml-pipeline/actions/runs/36109637276)
