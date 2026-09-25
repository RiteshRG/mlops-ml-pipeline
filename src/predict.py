from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/model.joblib")

EXPECTED_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

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