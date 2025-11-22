import pandas as pd
import numpy as np

# Load both datasets with absolute paths
phishing_df = pd.read_csv('D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/phishing.csv')
legit_df = pd.read_csv('D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/kaggle_legit.csv')

# Create a simple combined dataset
# Take a few samples from each dataset
phishing_sample = phishing_df.head(10).copy()
legit_sample = legit_df.head(10).copy()

# Create URL column for phishing data
phishing_sample['url'] = 'http://fake-payment-gateway.com'

# Create URL column for legit data
legit_sample['url'] = legit_sample['MerchantID'].apply(lambda x: f"https://legit-{x}.com")

# Combine relevant columns
phishing_data = pd.DataFrame({
    'url': phishing_sample['url'],
    'TransactionAmount': np.random.uniform(0.1, 100.0, len(phishing_sample)),
    'CustomerAge': np.random.randint(18, 80, len(phishing_sample)),
    'AccountBalance': np.random.uniform(100.0, 10000.0, len(phishing_sample)),
    'Location': np.random.choice(['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna'], len(phishing_sample)),
    'label': 1  # Fake gateway
})

legit_data = pd.DataFrame({
    'url': legit_sample['url'],
    'TransactionAmount': legit_sample['TransactionAmount'],
    'CustomerAge': legit_sample['CustomerAge'],
    'AccountBalance': legit_sample['AccountBalance'],
    'Location': legit_sample['Location'],
    'label': 0  # Legit gateway
})

# Combine datasets
combined_df = pd.concat([phishing_data, legit_data], ignore_index=True)

# Save combined dataset
combined_df.to_csv('D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/combined_dataset.csv', index=False)

print("Combined dataset created with shape:", combined_df.shape)
print("Label distribution:")
print(combined_df['label'].value_counts())
print("\nFirst few rows:")
print(combined_df.head(10))