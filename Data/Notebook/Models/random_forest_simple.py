#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Random Forest implementation for fake payment gateway detection
This is a streamlined version that's easy to run and understand
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os

def load_and_prepare_data(dataset_path='../../Processed/combined_dataset.csv'):
    """Load and prepare the feature data for Random Forest model training"""
    print("Current working directory:", os.getcwd())
    print("Attempting to load dataset from:", dataset_path)
    print("File exists:", os.path.exists(dataset_path))
    
    # Load the combined dataset
    df = pd.read_csv(dataset_path)
    
    # Feature engineering
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_special_chars'] = df['url'].apply(lambda x: sum(not c.isalnum() for c in x))
    df['has_ip'] = df['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
    df['url_entropy'] = df['url'].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Location_encoded'] = label_encoder.fit_transform(df['Location'])
    
    # Select features for modeling
    feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                      'url_length', 'num_dots', 'has_https', 'num_digits', 
                      'num_special_chars', 'has_ip', 'url_entropy', 'Location_encoded']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, df, label_encoder, feature_columns

def train_random_forest_model(X, y):
    """Train a Random Forest model with optimized parameters"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Optimized parameters
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    
    # Create and train Random Forest model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, accuracy, precision, recall, f1

def display_results(accuracy, precision, recall, f1):
    """Display model performance metrics"""
    print("\n" + "="*50)
    print("RANDOM FOREST MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

def display_feature_importance(model, feature_columns, top_n=10):
    """Display feature importance from the Random Forest model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance Rankings:")
    print("-" * 40)
    for i in range(min(top_n, len(feature_columns))):
        print(f"{i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")

def save_model(model, filepath='random_forest_model_simple.pkl'):
    """Save the trained model to disk"""
    joblib.dump(model, filepath)
    print(f"\nModel saved successfully as '{filepath}'")

def main():
    """Main function to run the simple Random Forest implementation"""
    print("Fake Payment Gateway Detection - Simple Random Forest Implementation")
    print("=" * 70)
    
    try:
        # Load and prepare data
        X, y, df, label_encoder, feature_columns = load_and_prepare_data()
        print(f"Dataset shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Train the model
        print("\nTraining Random Forest model...")
        model, X_test, y_test, y_pred, accuracy, precision, recall, f1 = train_random_forest_model(X, y)
        
        # Display results
        display_results(accuracy, precision, recall, f1)
        
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"{cm}")
        
        # Display feature importance
        display_feature_importance(model, feature_columns)
        
        # Save the trained model
        save_model(model, 'random_forest_model_simple.pkl')
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()