import pandas as pd
import numpy as np

# Load both datasets
phishing_df = pd.read_csv('../../../Data/Processed/phishing.csv')
legit_df = pd.read_csv('../../../Data/Processed/kaggle_legit.csv')

# Select relevant columns from phishing data
phishing_features = phishing_df[['UserId', 'Status', 'Mobile', 'ProgramId', 'BadTryCount', 'label']].copy()
phishing_features.columns = ['TransactionID', 'label', 'AccountID', 'CustomerAge', 'AccountBalance', 'dummy']
phishing_features['TransactionAmount'] = np.random.uniform(0.1, 100.0, len(phishing_features))
phishing_features['Location'] = 'Unknown'
phishing_features = phishing_features[['TransactionID', 'AccountID', 'TransactionAmount', 'CustomerAge', 'AccountBalance', 'Location', 'label']]

# Select relevant columns from legit data
legit_features = legit_df[['TransactionID', 'AccountID', 'TransactionAmount', 'CustomerAge', 'AccountBalance', 'Location', 'label']].copy()

# Combine datasets
combined_df = pd.concat([phishing_features, legit_features], ignore_index=True)

# Add URL column (simplified)
combined_df['url'] = combined_df['TransactionID'].apply(lambda x: f"http://example-{x}.com")

# Reorder columns
combined_df = combined_df[['url', 'TransactionAmount', 'CustomerAge', 'AccountBalance', 'Location', 'label']]

# Save combined dataset
combined_df.to_csv('../../../Data/Processed/combined_dataset.csv', index=False)

print("Combined dataset created with shape:", combined_df.shape)
print("Label distribution:")
print(combined_df['label'].value_counts())
print("\nFirst few rows:")
print(combined_df.head())