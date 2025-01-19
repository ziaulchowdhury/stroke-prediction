#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 01:27:49 2025

@author: Ziaul Chowdhury
"""

from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from hf_utils import push_model_to_hub

class TrainModel:
    
    def __init__(self, data_loader: DataLoader, data_preprocessor: DataPreprocessor):
        self.data_loader = data_loader
        self.data_preprocessor = data_preprocessor
        
        self.create__train_rfc_model()
        self.evaluate_rfc_model()
        self.save_model_locally()
        self.push_model_to_huggingface()
    
    
    def create__train_rfc_model(self):
        self.rfc_model = RandomForestClassifier(random_state=42)
        self.rfc_model.fit(self.data_preprocessor.X_train, self.data_preprocessor.y_train) 


    def evaluate_rfc_model(self):
        y_pred = self.rfc_model.predict(self.data_preprocessor.X_test)
        print("Classification Report:")
        print(classification_report(self.data_preprocessor.y_test, y_pred))


    def save_model_locally(self):
        joblib.dump(self.rfc_model, "../model/random_forest_model.pkl")
        print("Model saved locally.")
        
        
    def push_model_to_huggingface(self):
        push_model_to_hub(local_dir="../model", repo_name="zichowdhury/stroke-prediction", commit_message="Initial model upload.")


if __name__ == "__main__":
    file_path = "../data/healthcare-dataset-stroke-data.csv"
    data_loader = DataLoader(file_path)
    data_preprocessor = DataPreprocessor(data_loader)
    train_model = TrainModel(data_loader, data_preprocessor)
    
