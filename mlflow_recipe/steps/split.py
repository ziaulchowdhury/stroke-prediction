"""
Created on Sat Jan 18 11:39:22 2025

@author: Ziaul Chowdhury
"""
from pandas import DataFrame, Series

def create_dataset_filter(dataset: DataFrame) -> Series(bool):
    return Series(True, index=dataset.index)
