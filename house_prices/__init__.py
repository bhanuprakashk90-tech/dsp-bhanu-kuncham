from pathlib import Path

CONTINUOUS_FEATURES = [
    'LotArea', 'YearBuilt', 'OverallQual', 'OverallCond',
    'GrLivArea', 'TotalBsmtSF', 'GarageArea', '1stFlrSF'
]
CATEGORICAL_FEATURES = [
    'MSZoning', 'Neighborhood', 'BldgType',
    'HouseStyle', 'SaleCondition', 'SaleType'
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = 'SalePrice'

MODELS_DIR = Path(__file__).parent.parent / 'models'
