#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 01:41:17 2025

@author: Ziaul Chowdhury
"""

from huggingface_hub import HfApi, HfFolder, Repository, push_to_hub_keras
import os

def authenticate_huggingface(token: str):
    HfFolder.save_token(token)
    print("Authenticated with Hugging Face.")
    

def push_model_to_hub(local_dir: str, repo_name: str, commit_message: str):
    repo = Repository(local_dir=local_dir, clone_from=repo_name)
    repo.git_add(".")
    repo.git_commit(commit_message)
    repo.git_push()
    print(f"Model pushed to {repo_name} on Hugging Face Hub.")
