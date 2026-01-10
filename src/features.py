import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class PenguinFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer to calculate BMI and Bill Ratio 
    for the Palmer Penguins dataset.

    Assumes input DataFrame has 'body_mass_g', 'flipper_length_mm', 
    'bill_length_mm', and 'bill_depth_mm' columns.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer is stateless (doesn't learn parameters from data),
        # so we just return self.
        return self

    def transform(self, X, y=None):
        """
        Calculates BMI and Bill Ratio and adds them as new columns.
        
        BMI calculation uses the formula: 
        body_mass_g / (flipper_length_mm/100)^2 (mass in grams, length in meters)
        
        Bill Ratio calculation: bill_length_mm / bill_depth_mm
        """
        # Ensure X is a pandas DataFrame for column operations
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Make a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()
        
        # Calculate BMI (convert flipper length from mm to meters for realistic BMI scale)
        X_transformed['bmi'] = X_transformed['body_mass_g'] / (X_transformed['flipper_length_mm'] / 100)**2
        
        # Calculate Bill Ratio
        X_transformed['bill_ratio'] = X_transformed['bill_length_mm'] / X_transformed['bill_depth_mm']
        
        return X_transformed
    
# Get column names from the dataset once and derive feature columns (exclude target)
all_columns = load_penguins().columns.tolist()
target_column = "species"
feature_columns = [col for col in all_columns if col != target_column]

# Identify categorical columns (typically: island, sex) and remaining numerical columns
categorical_cols = [col for col in feature_columns if col in ['island', 'sex']]
numerical_cols = [col for col in feature_columns if col not in categorical_cols]

# Create preprocessing pipeline that only expects feature columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    verbose_feature_names_out=False,
    remainder="passthrough",  # keep engineered features that are added upstream
)
