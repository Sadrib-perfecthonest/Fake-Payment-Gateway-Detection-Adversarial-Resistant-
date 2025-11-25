import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random

def generate_enhanced_dataset(base_dataset_path='../../Processed/combined_dataset.csv'):
    """Generate an enhanced dataset with additional adversarial examples"""
    # Load the base dataset
    df = pd.read_csv(base_dataset_path)
    
    # Feature engineering for original data
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_special_chars'] = df['url'].apply(lambda x: sum(not c.isalnum() for c in x))
    df['has_ip'] = df['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
    df['url_entropy'] = df['url'].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Location_encoded'] = le.fit_transform(df['Location'])
    
    # Generate adversarial examples for phishing URLs
    phishing_df = df[df['label'] == 1].copy()
    legitimate_df = df[df['label'] == 0].copy()
    
    # Create adversarial examples by modifying phishing URLs
    adversarial_examples = []
    for idx, row in phishing_df.iterrows():
        # Create 2 adversarial examples for each phishing URL
        for i in range(2):
            # Modify URL slightly to create adversarial example
            modified_url = row['url'].replace('.', '' if i == 0 else '-', 1)
            adversarial_examples.append({
                'url': modified_url,
                'TransactionAmount': row['TransactionAmount'] * random.uniform(0.8, 1.2),
                'CustomerAge': row['CustomerAge'],
                'AccountBalance': row['AccountBalance'] * random.uniform(0.9, 1.1),
                'Location': row['Location'],
                'label': 1  # Still phishing
            })
    
    # Convert adversarial examples to DataFrame
    adversarial_df = pd.DataFrame(adversarial_examples)
    
    # Apply same feature engineering to adversarial examples
    adversarial_df['url_length'] = adversarial_df['url'].apply(len)
    adversarial_df['num_dots'] = adversarial_df['url'].apply(lambda x: x.count('.'))
    adversarial_df['has_https'] = adversarial_df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    adversarial_df['num_digits'] = adversarial_df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    adversarial_df['num_special_chars'] = adversarial_df['url'].apply(lambda x: sum(not c.isalnum() for c in x))
    adversarial_df['has_ip'] = adversarial_df['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
    adversarial_df['url_entropy'] = adversarial_df['url'].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
    
    # Encode categorical variables for adversarial examples
    adversarial_df['Location_encoded'] = le.transform(adversarial_df['Location'])
    
    # Combine all data
    enhanced_df = pd.concat([df, adversarial_df], ignore_index=True)
    
    return enhanced_df

def save_enhanced_dataset(enhanced_df, output_path='../../Processed/enhanced_combined_dataset.csv'):
    """Save the enhanced dataset to a CSV file"""
    # Select original columns for saving
    original_columns = ['url', 'TransactionAmount', 'CustomerAge', 'AccountBalance', 'Location', 'label']
    enhanced_df[original_columns].to_csv(output_path, index=False)
    print(f"Enhanced dataset saved to {output_path}")

def main():
    """Main function to generate and save the enhanced dataset"""
    print("Fake Payment Gateway Detection - Enhanced Dataset Generator")
    print("=" * 65)
    
    try:
        # Generate enhanced dataset
        enhanced_df = generate_enhanced_dataset()
        print(f"Original dataset size: {len(enhanced_df) - len(enhanced_df[enhanced_df.duplicated(subset=['url'], keep='first')])}")
        print(f"Enhanced dataset size: {len(enhanced_df)}")
        print("Label distribution:")
        print(enhanced_df['label'].value_counts())
        
        # Save enhanced dataset
        save_enhanced_dataset(enhanced_df)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()