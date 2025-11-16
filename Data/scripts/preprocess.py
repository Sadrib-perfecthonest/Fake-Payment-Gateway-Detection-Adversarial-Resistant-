import pandas as pd
import tldextract
import os

def extract_domain_features(url):

    ext = tldextract.extract(url)

    return ext.domain, ext.suffix

print("--- Running Preprocessing Pipeline ---")


try:
    phish = pd.read_csv("../raw/phishtank_urls.csv")
    legit = pd.read_csv("../processed/kaggle_legit.csv") 

    manual_data = pd.read_csv("../raw/adverserial_legit.csv") 
except FileNotFoundError as e:
    print(f"Error: Required file not found. Have you run collect_kaggle.py and collect_phishtank.py? Error: {e}")
    exit()


phish['label'] = 1  
phish_cols = ['url', 'label']


transaction_cols = ['TransactionAmount', 'TransactionType', 'CustomerAge', 'AccountBalance', 'Location']
for col in transaction_cols:
    phish[col] = None 
    
phish_df = phish[phish_cols + transaction_cols]


kaggle_df = legit[phish_cols + transaction_cols] 


manual_df = manual_data[phish_cols + transaction_cols]


df = pd.concat([
    phish_df, 
    kaggle_df,
    manual_df
], ignore_index=True).drop_duplicates(subset=['url']) 

print(f"Total training dataset size after merge: {len(df)}")

print("Extracting domain and TLD...")

df[['domain', 'tld']] = df['url'].apply(lambda u: pd.Series(extract_domain_features(u)))

output_path = "../processed/preprocessed_dataset.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
df.to_csv(output_path, index=False)

print(f"Saved cleaned TRAINING dataset to {output_path}")