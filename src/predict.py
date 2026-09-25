from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/model.joblib")

EXPECTED_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

def predict(sample: dict):
    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in sample
    ]

    if missing_features:
        raise ValueError(
            f"Missing required feature(s): {missing_features}"
        )

    model = joblib.load(MODEL_PATH)

    input_data = pd.DataFrame(
        [[sample[feature]for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES
    )

    prediction = model.predict(input_data)

    return prediction