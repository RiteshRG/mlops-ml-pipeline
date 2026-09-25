from pathlib import Path
import Pandas as pd
import sys

DATA_PATH = Path("data/breast_cancer_wisconsin.csv")

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

def validate_dataset():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    required_columns = EXPECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print(f"ERROR: Missing required columns:")
        for col in missing_columns:
            print(f" - {col}")
        sys.exit(1)

    if df.empty:
        print(f"ERROR: Dataset is empty")
        sys.exit(1)

    if df[TARGET_COLUMN].isnull().any():
        print("ERROR: Target column contains missing values.")
        sys.exit(1)

    print("Dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(EXPECTED_FEATURES)}")
    print(f"Target: {TARGET_COLUMN}")

if __name__ == "__main__":
    validate_dataset()