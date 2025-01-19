#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:35:44 2025

@author: zic
"""

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load dataset
file_path = "../data/healthcare-dataset-stroke-data.csv"  # Update with your dataset path
data = pd.read_csv(file_path)

# Data preprocessing
# Fill missing values in 'bmi' with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# Drop irrelevant column
if 'id' in data.columns:
    data.drop(columns=['id'], inplace=True)

# Split features and target
X = data.drop(columns=['stroke'])
y = data['stroke']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Train and evaluate Logistic Regression
print("Logistic Regression Results:")
lr_pipeline.fit(X_train, y_train)
evaluate_model(lr_pipeline, X_test, y_test)

# Train and evaluate Random Forest
print("Random Forest Results:")
rf_pipeline.fit(X_train, y_train)
evaluate_model(rf_pipeline, X_test, y_test)

# Interpret Random Forest with Feature Importances
rf_model = rf_pipeline.named_steps['classifier']
feature_names = list(X.select_dtypes(include=['float64', 'int64']).columns) + \
                list(rf_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_cols))
feature_importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

print("\nTop Features in Random Forest:")
print(feature_importances.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.head(10), y=feature_importances.head(10).index, palette="viridis")
plt.title('Top 10 Feature Importances in Random Forest')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.show()

# SHAP Explanation (for Random Forest)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(preprocessor.transform(X_test))

# Summary plot
shap.summary_plot(shap_values[1], preprocessor.transform(X_test), feature_names=feature_names)
