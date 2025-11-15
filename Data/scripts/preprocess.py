import pandas as pd
import tldextract

def extract_domain_features(url):
    ext = tldextract.extract(url)
    return ext.domain, ext.suffix

print("Loading raw datasets...")
phish = pd.read_csv("../data/raw/phishtank_urls.csv")
legit = pd.read_csv("../data/raw/kaggle_legit.csv")

phish['label'] = 1
legit['label'] = 0

df = pd.concat([phish[['url','label']], legit[['url','label']]], ignore_index=True)

print("Extracting domain and TLD...")
df[['domain', 'tld']] = df['url'].apply(lambda u: pd.Series(extract_domain_features(u)))

df.to_csv("../data/processed/sessions_clean.csv", index=False)
print("Saved cleaned dataset.")
