import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from house_prices import FEATURE_COLUMNS, MODELS_DIR
from house_prices.preprocess import preprocess


def _load_model() -> LinearRegression:
    """Load trained model from disk.

    Returns:
        Fitted LinearRegression model.

    Raises:
        FileNotFoundError: If model artifact doesn't exist.
    """
    model_path = MODELS_DIR / 'model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def _load_input(filepath: str) -> pd.DataFrame:
    """Load and validate inference input data from CSV file.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Dataframe with required feature columns.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    df = pd.read_csv(filepath)
    return df[FEATURE_COLUMNS]


def make_predictions(filepath: str) -> np.ndarray:
    """Load model and return predictions for input data.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Array of predicted house prices.
    """
    model = _load_model()
    X = _load_input(filepath)
    X_processed = preprocess(X, is_training=False)
    predictions = model.predict(X_processed)
    return np.maximum(predictions, 1)
