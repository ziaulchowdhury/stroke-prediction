"""
Created on Sat Jan 18 11:39:34 2025

@author: Ziaul Chowdhury
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def transformer_fn() -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Combine the transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor)
        ]
    )
