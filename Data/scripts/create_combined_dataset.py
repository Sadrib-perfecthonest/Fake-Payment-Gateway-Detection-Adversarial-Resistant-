import pandas as pd
import os

# Load the datasets
phishing_df = pd.read_csv('Data/Processed/phishing.csv')
legit_df = pd.read_csv('Data/Processed/kaggle_legit.csv')

print("Phishing dataset columns:", phishing_df.columns.tolist())
print("Legit dataset columns:", legit_df.columns.tolist())

# Create a combined dataset with the structure expected by the model
# We'll need to create the required columns for both datasets

# For the legitimate dataset, create the required structure
legit_combined = pd.DataFrame({
    'url': legit_df['MerchantID'].apply(lambda x: f"https://www.{x.lower()}.com" if pd.notnull(x) else "https://www.merchant.com"),
    'TransactionAmount': legit_df['TransactionAmount'],
    'CustomerAge': legit_df['CustomerAge'],
    'AccountBalance': legit_df['AccountBalance'],
    'Location': legit_df['Location'],
    'label': legit_df['label']
})

# For the phishing dataset, we need to create the missing columns
# Since the phishing dataset doesn't have the required financial columns, we'll add default values
phishing_combined = pd.DataFrame({
    'url': phishing_df['Email'].apply(lambda x: f"https://www.{x.split('@')[1]}" if pd.notnull(x) and '@' in str(x) else "https://www.suspicious-site.com"),
    'TransactionAmount': 100.0,  # Default value
    'CustomerAge': 30,  # Default value
    'AccountBalance': 5000.0,  # Default value
    'Location': 'Unknown',  # Default value
    'label': phishing_df['label']
})

# Combine both datasets
combined_df = pd.concat([legit_combined, phishing_combined], ignore_index=True)

# Save the combined dataset
output_path = 'Data/Processed/combined_dataset.csv'
combined_df.to_csv(output_path, index=False)

print(f"Combined dataset saved to {output_path}")
print(f"Combined dataset shape: {combined_df.shape}")
print("Label distribution:")
print(combined_df['label'].value_counts())

# Show first few rows
print("\nFirst 5 rows of combined dataset:")
print(combined_df.head())