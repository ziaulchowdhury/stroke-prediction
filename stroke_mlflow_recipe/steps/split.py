"""
Created on Sat Jan 18 11:39:22 2025

@author: Ziaul Chowdhury
"""

import pandas as pd
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_dataset_filter(dataset: DataFrame) -> Series(bool):
    return Series(True, index=dataset.index)

def custom_split(df: DataFrame):
    print(f'Columns: {df.columns}')
    df.drop(columns=['id'], inplace=True)
    print(f'Dropped ID column. Updated columns: {df.columns}')

    # Fill missing BMI values with mean
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    print(f"Number of positive labels in stroke dataset:{df[df['stroke']==1]}")
    print(f"Number of negative labels in stroke dataset:{df[df['stroke'] == 0]}")


    # df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_ratio = 0.8
    test_ratio = 0.1

    total_rows = len(df)
    train_end = int(total_rows * train_ratio)
    test_end = train_end + int(total_rows * test_ratio)

    splits = pd.Series(index=df.index, dtype="object")
    splits.iloc[:train_end] = "TRAINING"
    splits.iloc[train_end:test_end] = "TEST"
    splits.iloc[test_end:] = "VALIDATION"

    return splits
