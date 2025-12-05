# Basic tests for predict function

import os
import sys
from pathlib import Path

# Ensure app package is on path when running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.predict import predict


def test_predict_returns_known_classes():
    # Using three known Iris samples the model was trained on
    assert predict([5.1, 3.5, 1.4, 0.2]) == "setosa"
    assert predict([6.0, 2.9, 4.5, 1.5]) == "versicolor"
    assert predict([6.9, 3.1, 5.4, 2.1]) == "virginica"


def test_predict_class_is_valid_label():
    # The model predicts one of the three Iris species labels
    valid_labels = {"setosa", "versicolor", "virginica"}
    label = predict([5.9, 3.0, 5.1, 1.8])
    assert label in valid_labels
