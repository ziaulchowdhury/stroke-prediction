"""
Created on Sat Jan 18 11:39:44 2025
@author: Ziaul Chowdhury
"""
from typing import Dict, Any

def estimator_fn(estimator_params: Dict[str, Any] = None):
    from sklearn.linear_model import SGDClassifier

    if estimator_params is None:
        estimator_params = {}
    return SGDClassifier(random_state=42, **estimator_params)