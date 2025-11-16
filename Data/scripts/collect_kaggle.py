import pandas as pd
import random

input_path = "../raw/bank_transactions_data_kaggle.csv"
output_path = "../processed/kaggle_legit.csv"

bd_cities = ["Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Khulna"]

print("Loading Kaggle dataset...")
df = pd.read_csv(input_path)

df_small = df.head(14)  

df_small["Location"] = [random.choice(bd_cities) for _ in range(len(df_small))]
df_small["label"] = 0  

df_small.to_csv(output_path, index=False)

print("Saved Kaggle legitimate dataset to data/raw/kaggle_legit.csv")
