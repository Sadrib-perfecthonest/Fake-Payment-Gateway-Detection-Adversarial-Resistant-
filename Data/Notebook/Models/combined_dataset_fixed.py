import pandas as pd
import numpy as np
import random
import string

# Load both datasets with absolute paths
phishing_df = pd.read_csv('d:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/phishing.csv')
legit_df = pd.read_csv('d:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/kaggle_legit.csv')

# Function to generate more realistic phishing URLs that closely mimic legitimate ones
def generate_phishing_url(base_url="paypal"):
    # Common patterns used by phishing sites
    prefixes = ["", "secure.", "login.", "account.", "verify.", "update.", "my."]
    suffixes = [".com", ".net", ".org", ".info", ".biz", ".co"]
    
    # Randomly choose variations
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    
    # Use HTTPS most of the time to make it more realistic
    protocol = "https"
    
    # Add some randomness to make it more challenging
    if random.random() > 0.7:
        # Add numbers
        num = str(random.randint(1, 999))
        return f"{protocol}://{prefix}{base_url}{num}{suffix}"
    elif random.random() > 0.7:
        # Add special characters (homograph attacks)
        homographs = {'a': '@', 'o': '0', 'i': '1', 'e': '3', 's': '5'}
        url_part = f"{prefix}{base_url}{suffix}"
        for char, homograph in homographs.items():
            if random.random() > 0.8:  # Low probability to avoid making it too obvious
                url_part = url_part.replace(char, homograph)
        return f"{protocol}://{url_part}"
    else:
        # Regular variation
        return f"{protocol}://{prefix}{base_url}{suffix}"

# Create phishing features with more variety
phishing_features = []
sample_size = min(50, len(phishing_df))

for i in range(sample_size):
    # Use different base names to make it more realistic
    base_names = ["paypal", "skrill", "stripe", "visa", "mastercard", "amex", "bankofamerica", "chase", "wellsfargo"]
    base_url = random.choice(base_names)
    url = generate_phishing_url(base_url)
    
    phishing_features.append({
        'url': url,
        'TransactionAmount': np.random.uniform(1.0, 500.0),  # Wider range
        'CustomerAge': np.random.randint(18, 80),
        'AccountBalance': np.random.uniform(100.0, 20000.0),  # Wider range
        'Location': np.random.choice(['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna', 'Rangpur', 'Barisal', 'Mymensingh']),
        'label': 1
    })

# Select legitimate data but make it more similar to phishing
legit_features = []
sample_size = min(50, len(legit_df))

for i in range(sample_size):
    row = legit_df.iloc[i]
    # Use actual transaction data but with similar structure to phishing
    legit_features.append({
        'url': f"https://secure.{row['TransactionID']}.com",  # Make it look similar to phishing
        'TransactionAmount': row['TransactionAmount'] if 'TransactionAmount' in legit_df.columns else np.random.uniform(10.0, 1000.0),
        'CustomerAge': row['CustomerAge'] if 'CustomerAge' in legit_df.columns else np.random.randint(18, 70),
        'AccountBalance': row['AccountBalance'] if 'AccountBalance' in legit_df.columns else np.random.uniform(1000.0, 15000.0),
        'Location': row['Location'] if 'Location' in legit_df.columns else np.random.choice(['Dhaka', 'Chittagong', 'Sylhet', 'Rajshahi', 'Khulna']),
        'label': 0
    })

# Combine datasets
combined_data = phishing_features + legit_features
combined_df = pd.DataFrame(combined_data)

# Save combined dataset
combined_df.to_csv('d:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/combined_dataset.csv', index=False)

print("Combined dataset created with shape:", combined_df.shape)
print("Label distribution:")
label_counts = combined_df['label'].value_counts()
print(label_counts)
print("\nSample URLs:")
print("Phishing examples:")
for i in range(5):
    print(f"  {combined_df[combined_df['label']==1].iloc[i]['url']}")
print("Legitimate examples:")
for i in range(5):
    print(f"  {combined_df[combined_df['label']==0].iloc[i]['url']}")

# Show feature distributions
print("\nHTTPS feature distribution:")
combined_df['has_https'] = combined_df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
print(combined_df.groupby(['label', 'has_https']).size().unstack(fill_value=0))

# Show URL length distribution
print("\nURL length distribution:")
combined_df['url_length'] = combined_df['url'].apply(len)
print("Phishing URL length mean:", combined_df[combined_df['label']==1]['url_length'].mean())
print("Legitimate URL length mean:", combined_df[combined_df['label']==0]['url_length'].mean())