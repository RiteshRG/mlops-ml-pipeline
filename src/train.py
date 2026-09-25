from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

DATA_PATH = Path("data/breast_cancer_wisconsin.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

TARGET_COLUMN = "diagnosis"

EXPECTED_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

TEST_SET = 0.20
RANDOM_STATE = 42

# Minimum required improvement over baseline
MARGIN = 0.05

def main():
    # 1. load dataset
    df = pd.read_csv(DATA_PATH)
    x_data = df[EXPECTED_FEATURES]
    y_data = df[TARGET_COLUMN]

    # 2. data split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=TEST_SET, random_state=RANDOM_STATE)

    # 3. train dummy classifier (Baseline)
    dummy_model = DummyClassifier(strategy="most_frequent")

    dummy_model.fit(x_train,y_train)

    baseline_predictions = dummy_model.predict(x_test)

    baseline_f1 = f1_score(y_test, baseline_predictions)

    # 4. train randomforestclassifier (Candidate model)
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    model_f1 = f1_score(y_test,predictions)

    # 5. Quality gate
    required_score = baseline_f1 + MARGIN

    quality_gate_passed = model_f1 >= required_score

    print(f"Baseline F1: {baseline_f1:.4f}")
    print(f"Random Forest F1: {model_f1:.4f}")
    print(f"Required F1: {required_score:.4f}")
    print(f"Margin: {MARGIN:.4f}")

    if not quality_gate_passed:
        print("QUALITY GET FAILED")
        print(
            "Random Forest did not improve enough over the baseline."
        )
        sys.exit(1)

    print("QUALITY GATE PASSED.")

    # 6. save model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    # 7. save metrics
    metrics = {
        "baseline_model": "DummyClassifier",
        "candidate_model": "RandomForestClassifier",
        "metric": "f1",
        "baseline_score": baseline_f1,
        "model_score": model_f1,
        "improvement_margin": MARGIN,
        "required_score": required_score,
        "quality_gate_passed": quality_gate_passed,
        "test_size": TEST_SET,
        "random_state": RANDOM_STATE
    }

    with open(METRICS_PATH, "w")as file:
        json.dump(metrics, file, indent=4)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")








if __name__ == "__main__":
    main()