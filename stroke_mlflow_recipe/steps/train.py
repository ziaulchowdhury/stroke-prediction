"""
Created on Sat Jan 18 11:39:44 2025
@author: Ziaul Chowdhury
"""
from typing import Dict, Any

def estimator_fn(estimator_params: Dict[str, Any] = None):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(random_state=42, **estimator_params)

def my_early_stop_fn(*args):
  from hyperopt.early_stop import no_progress_loss
  return no_progress_loss(10)(*args)