import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
from house_prices import FEATURE_COLUMNS, LABEL_COLUMN, MODELS_DIR
from house_prices.preprocess import preprocess


def _load_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate training data from CSV file.

    Args:
        filepath: Path to training CSV file.

    Returns:
        Tuple of (features dataframe, target series).

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Training file not found: {filepath}")
    df = pd.read_csv(filepath).dropna(subset=[LABEL_COLUMN])
    return df[FEATURE_COLUMNS], df[LABEL_COLUMN]


def _train_and_save(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Train linear regression model and save to disk.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / 'model.joblib')
    return model


def _compute_metrics(
    model: LinearRegression, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    """Compute RMSLE and R² metrics for model predictions.

    Args:
        model: Fitted LinearRegression model.
        X: Features for evaluation.
        y: Target values.

    Returns:
        Dictionary with 'rmsle' and 'r2' metrics.
    """
    y_pred = np.maximum(model.predict(X), 1)
    rmsle = round(float(np.sqrt(mean_squared_log_error(y, y_pred))), 4)
    r2 = round(float(r2_score(y, y_pred)), 4)
    return {'rmsle': rmsle, 'r2': r2}


def build_model(filepath: str) -> dict[str, float]:
    """Train model on data and return evaluation metrics.

    Args:
        filepath: Path to training CSV file.

    Returns:
        Dictionary with metric names as keys and values as floats.
    """
    X, y = _load_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_processed = preprocess(X_train, is_training=True)
    X_test_processed = preprocess(X_test, is_training=False)
    model = _train_and_save(X_train_processed, y_train)
    metrics = _compute_metrics(model, X_test_processed, y_test)
    return metrics
