import joblib
import pytest
import sys
from pathlib import Path

from src.predict import predict, MODEL_PATH, EXPECTED_FEATURES

def create_valid_sample():
    return{
        feature:1.0
        for feature in EXPECTED_FEATURES
    }

def test_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None

def test_valid_sample_prediction():
    sample = create_valid_sample()

    prediction = predict(sample)

    assert prediction.shape == (1,)

def test_missing_feature_is_rejected():
    sample = create_valid_sample()
    sample.pop(EXPECTED_FEATURES[0])
    with pytest.raises(ValueError, match="Missing required feature"):
        predict(sample)
