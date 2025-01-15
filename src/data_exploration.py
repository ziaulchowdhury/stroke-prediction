#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:24:18 2025

@author: Ziaul Chowdhury
"""

from data_loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt


class DataExplorer:
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.numerical_cols = self.data_loader.data.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_cols = self.data_loader.data.select_dtypes(include=['object']).columns
        
        self.describe_dataset()
        self.print_missing_values()
        self.explore_label_class_distribution()
        self.explore_categorical_features()
        self.explore_numerical_features()
        self.explore_correlations_numerical_features()
        self.explore_relationship_between_features_target_variable()
        self.explore_pairplot_numerical_features()
        
    def describe_dataset(self):
        print("First few rows of the dataset:")
        print(self.data_loader.data.head())

        print("Basic information about the dataset:")
        print(self.data_loader.data.info())

        print("Summary statistics of numerical features:")
        print(self.data_loader.data.describe())

        print("Summary statistics of categorical features:")
        print(self.data_loader.data.describe(include=['object']))

    def print_missing_values(self):
        print("Missing values per column:")
        print(self.data_loader.data.isnull().sum())

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.data_loader.data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values in Dataset')
        plt.show()

    def explore_label_class_distribution(self):
        print("Class distribution of the target variable ('stroke'):")
        print(self.data_loader.data['stroke'].value_counts(normalize=True))

        plt.figure(figsize=(6, 4))
        sns.countplot(x='stroke', data=self.data_loader.data, palette='pastel')
        plt.title('Class Distribution of Stroke')
        plt.xlabel('Stroke')
        plt.ylabel('Count')
        plt.show()
        
    def explore_categorical_features(self):
        print("Unique values in categorical columns:")
        for col in self.categorical_cols:
            print(f"{col}: {self.data_loader.data[col].unique()}")

        for col in self.categorical_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=col, data=self.data_loader.data, palette='coolwarm', 
                          order=self.data_loader.data[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

    def explore_numerical_features(self):
        print("Summary statistics of numerical columns:")
        print(self.data_loader.data[self.numerical_cols].describe())
        
        for col in self.numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data_loader.data[col], kde=True, color='blue', bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def explore_correlations_numerical_features(self):
        correlation_matrix = self.data_loader.data[self.numerical_cols].corr()
        print("Correlation matrix of numerical features:")
        print(correlation_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()
        
    def explore_relationship_between_features_target_variable(self):
        for col in self.numerical_cols:
            if col != 'stroke':
                plt.figure(figsize=(8, 4))
                sns.boxplot(x='stroke', y=col, data=self.data_loader.data, palette='Set2')
                plt.title(f"Relationship between {col} and Stroke")
                plt.xlabel('Stroke')
                plt.ylabel(col)
                plt.show()
                
    def explore_pairplot_numerical_features(self):
        if len(self.numerical_cols) > 2:
            sns.pairplot(self.data_loader.data, hue="stroke", vars=self.numerical_cols, palette='husl')
            plt.suptitle("Pairplot of Numerical Features")
            plt.show()

if __name__ == "__main__":
    
    file_path = "../data/healthcare-dataset-stroke-data.csv"
    data_loader = DataLoader(file_path)
    data_explorer = DataExplorer(data_loader)