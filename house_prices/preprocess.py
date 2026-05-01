import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR


def _fit_save_transformers(
    df: pd.DataFrame,
) -> tuple[SimpleImputer, SimpleImputer, StandardScaler, OneHotEncoder]:
    """Fit and save imputers, scaler, and encoder to disk."""
    imputer_cont = SimpleImputer(strategy='mean')
    df_cont = pd.DataFrame(
        imputer_cont.fit_transform(df[CONTINUOUS_FEATURES]),
        columns=CONTINUOUS_FEATURES,
        index=df.index
    )
    joblib.dump(imputer_cont, MODELS_DIR / 'imputer_cont.joblib')

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_cat = pd.DataFrame(
        imputer_cat.fit_transform(df[CATEGORICAL_FEATURES]),
        columns=CATEGORICAL_FEATURES,
        index=df.index
    )
    joblib.dump(imputer_cat, MODELS_DIR / 'imputer_cat.joblib')

    scaler = StandardScaler()
    scaler.fit(df_cont)
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df_cat)
    joblib.dump(encoder, MODELS_DIR / 'encoder.joblib')

    return imputer_cont, imputer_cat, scaler, encoder


def _load_transformers() -> tuple[
    SimpleImputer, SimpleImputer, StandardScaler, OneHotEncoder
]:
    """Load imputers, scaler, and encoder from disk."""
    files = [
        'imputer_cont.joblib',
        'imputer_cat.joblib',
        'scaler.joblib',
        'encoder.joblib'
    ]
    for name in files:
        if not (MODELS_DIR / name).exists():
            raise FileNotFoundError(f"{name} not found. Run build_model().")

    imputer_cont = joblib.load(MODELS_DIR / 'imputer_cont.joblib')
    imputer_cat = joblib.load(MODELS_DIR / 'imputer_cat.joblib')
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    encoder = joblib.load(MODELS_DIR / 'encoder.joblib')

    return imputer_cont, imputer_cat, scaler, encoder


def _apply_transformers(
    df: pd.DataFrame,
    imputer_cont: SimpleImputer,
    imputer_cat: SimpleImputer,
    scaler: StandardScaler,
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """Apply fitted transformers to features."""
    df_cont = pd.DataFrame(
        imputer_cont.transform(df[CONTINUOUS_FEATURES]),
        columns=CONTINUOUS_FEATURES,
        index=df.index
    )
    df_cat = pd.DataFrame(
        imputer_cat.transform(df[CATEGORICAL_FEATURES]),
        columns=CATEGORICAL_FEATURES,
        index=df.index
    )

    scaled = pd.DataFrame(
        scaler.transform(df_cont),
        columns=CONTINUOUS_FEATURES,
        index=df.index
    )
    encoded = pd.DataFrame(
        encoder.transform(df_cat),
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
    if is_training:
        i_cont, i_cat, scaler, encoder = _fit_save_transformers(df)
    else:
        i_cont, i_cat, scaler, encoder = _load_transformers()
    return _apply_transformers(df, i_cont, i_cat, scaler, encoder)
