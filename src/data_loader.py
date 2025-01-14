#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 01:23:45 2025

@author: zic
"""
import pandas as pd

class DataLoader:
    
    def __init__(self, file_path="../data/healthcare-dataset-stroke-data.csv"):
        self.file_path = file_path
        self.load_file()
        
    def load_file(self):
        self.data = pd.read_csv(self.file_path)