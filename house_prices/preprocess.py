import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with mean for continuous and categorical features.

    Args:
        df: Input dataframe with potential missing values.

    Returns:
        Dataframe with missing values filled.
    """
    df = df.copy()
    for col in CONTINUOUS_FEATURES:
        df[col] = df[col].fillna(df[col].mean())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def _fit_save_transformers(
    df: pd.DataFrame,
) -> tuple[StandardScaler, OneHotEncoder]:
    """Fit and save scaler and encoder to disk.

    Args:
        df: Training dataframe for fitting transformers.

    Returns:
        Tuple of (fitted scaler, fitted encoder).
    """
    scaler = StandardScaler()
    scaler.fit(df[CONTINUOUS_FEATURES])
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[CATEGORICAL_FEATURES])
    joblib.dump(encoder, MODELS_DIR / 'encoder.joblib')
    return scaler, encoder


def _load_transformers() -> tuple[StandardScaler, OneHotEncoder]:
    """Load scaler and encoder from disk.

    Returns:
        Tuple of (loaded scaler, loaded encoder).

    Raises:
        FileNotFoundError: If scaler or encoder artifacts don't exist.
    """
    scaler_path = MODELS_DIR / 'scaler.joblib'
    if not scaler_path.exists():
        raise FileNotFoundError("Scaler not found. Run build_model() first.")
    scaler = joblib.load(scaler_path)

    encoder_path = MODELS_DIR / 'encoder.joblib'
    if not encoder_path.exists():
        raise FileNotFoundError("Encoder not found. Run build_model() first.")
    encoder = joblib.load(encoder_path)

    return scaler, encoder


def _apply_transformers(
    df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder
) -> pd.DataFrame:
    """Apply fitted scaler and encoder to features.

    Args:
        df: Input dataframe to transform.
        scaler: Fitted StandardScaler for continuous features.
        encoder: Fitted OneHotEncoder for categorical features.

    Returns:
        Dataframe with scaled and encoded features.
    """
    scaled = pd.DataFrame(
        scaler.transform(df[CONTINUOUS_FEATURES]),
        columns=CONTINUOUS_FEATURES,
        index=df.index
    )
    encoded = pd.DataFrame(
        encoder.transform(df[CATEGORICAL_FEATURES]),
        columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        index=df.index
    )

    result = pd.concat([scaled, encoded], axis=1)
    return result


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Preprocess features for training or inference.

    Args:
        df: Raw dataframe with feature columns.
        is_training: If True, fit and save transformers.
            If False, load saved transformers.

    Returns:
        Preprocessed dataframe ready for model.
    """
    df = _fill_missing(df)
    if is_training:
        scaler, encoder = _fit_save_transformers(df)
    else:
        scaler, encoder = _load_transformers()
    return _apply_transformers(df, scaler, encoder)
