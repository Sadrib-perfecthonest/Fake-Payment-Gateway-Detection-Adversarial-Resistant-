import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data_fixed():
    """Load and prepare the feature data for model training with fixed preprocessing"""
    # Load the combined dataset
    df = pd.read_csv('../../Processed/combined_dataset.csv')
    
    # Feature engineering
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_special_chars'] = df['url'].apply(lambda x: sum(not c.isalnum() for c in x))
    df['has_ip'] = df['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Location_encoded'] = le.fit_transform(df['Location'])
    
    # Select features for modeling
    feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                      'url_length', 'num_dots', 'has_https', 'num_digits', 
                      'num_special_chars', 'has_ip', 'Location_encoded']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, df

def main():
    """Main function to demonstrate the fixed dataset preparation"""
    print("Fake Payment Gateway Detection - Fixed Dataset Preparation")
    print("=" * 60)
    
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data_fixed()
        print(f"Dataset shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        print(f"\nTraining set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()