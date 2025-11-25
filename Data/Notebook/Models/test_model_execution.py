#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify model execution
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Print current working directory
print("Current working directory:", os.getcwd())

# Check if the dataset file exists
dataset_path = '../../Processed/combined_dataset.csv'
print("Dataset file exists:", os.path.exists(dataset_path))

if os.path.exists(dataset_path):
    print("Absolute path to dataset:", os.path.abspath(dataset_path))
    
    # Try to load the dataset
    try:
        df = pd.read_csv(dataset_path)
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        
        # Try to import and run the model
        try:
            from Data.Notebook.Models.random_forest_simple import train_random_forest_model, load_and_prepare_data
            
            print("\n" + "="*50)
            print("Initializing Random Forest Model...")
            print("="*50)
            
            # Load and prepare data
            X, y, df_processed, label_encoder, feature_columns = load_and_prepare_data(dataset_path)
            print(f"Processed dataset shape: {df_processed.shape}")
            print("Label distribution:")
            print(df_processed['label'].value_counts())
            
            print("\nData preparation successful!")
            
        except Exception as e:
            print(f"Error importing or initializing model: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Dataset file not found at: {dataset_path}")
    print("Files in Data/Processed directory:")
    if os.path.exists('../../Processed'):
        for file in os.listdir('../../Processed'):
            print(f"  {file}")