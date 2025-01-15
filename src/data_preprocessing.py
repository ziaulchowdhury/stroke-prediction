#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:30:02 2025

@author: Ziaul Chowdhury
"""

from data_loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        
        self.print_missing_values()
        self.fill_missing_bmi_with_mean()
        self.drop_non_predictive_columns()
        self.separate_input_output()
        self.create_preprocessor_pipeline_for_features()
        self.split_train_test_and_transform_data()
        
    
    def print_missing_values(self):
        print("Missing values per column:")
        print(self.data_loader.data.isnull().sum())

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data_loader.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values in Dataset')
        plt.show() 
        
        
    def fill_missing_bmi_with_mean(self):
        self.data_loader.data['bmi'] = self.data_loader.data['bmi'].fillna(self.data_loader.data['bmi'].mean())


    def drop_non_predictive_columns(self):
        if 'id' in self.data_loader.data.columns:
            self.data_loader.data.drop(columns=['id'], inplace=True)
            
    def separate_input_output(self):
        self.X = self.data_loader.data.drop(columns=['stroke'])
        self.y = self.data_loader.data['stroke']
        
    def create_preprocessor_pipeline_for_features(self):
        categorical_cols = self.X.select_dtypes(include=['object']).columns
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns

        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
    def split_train_test_and_transform_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_test = self.preprocessor.transform(self.X_test)
        print(f"Transformed training features shape: {self.X_train.shape}")
        print(f"Transformed testing features shape: {self.X_test.shape}")


if __name__ == "__main__":
    
    file_path = "../data/healthcare-dataset-stroke-data.csv"
    data_loader = DataLoader(file_path)
    data_preprocessor = DataPreprocessor(data_loader)