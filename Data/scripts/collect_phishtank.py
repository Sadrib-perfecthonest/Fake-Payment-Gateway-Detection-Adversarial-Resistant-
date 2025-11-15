import pandas as pd
import requests


url = "http://data.phishtank.com/data/online-valid.csv"

print("Downloading PhishTank dataset...")
df = pd.read_csv(url)

df[['url', 'phish_id']].to_csv("../data/raw/phishtank_urls.csv", index=False)

print("Saved to data/raw/phishtank_urls.csv")
