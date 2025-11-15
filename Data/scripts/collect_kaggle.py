import pandas as pd

# Replace this path with your Kaggle downloaded file
input_path = "../data/raw/kaggle_original.csv"

print("Loading Kaggle legitimate URLs...")
df = pd.read_csv(input_path)

df.to_csv("../data/raw/kaggle_legit.csv", index=False)

print("Saved to data/raw/kaggle_legit.csv")
