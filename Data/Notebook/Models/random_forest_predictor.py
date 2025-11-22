import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class RandomForestPredictor:
    """A class to use the trained Random Forest model for predictions"""
    
    def __init__(self, model_path, label_encoder=None):
        """Initialize the predictor with a trained model"""
        self.model = joblib.load(model_path)
        self.label_encoder = label_encoder
        
    def engineer_features(self, url, transaction_amount, customer_age, account_balance, location):
        """Engineer features for a single prediction"""
        # Feature engineering (same as training)
        url_length = len(url)
        num_dots = url.count('.')
        has_https = 1 if url.startswith('https') else 0
        num_digits = sum(c.isdigit() for c in url)
        num_special_chars = sum(not c.isalnum() for c in url)
        has_ip = 1 if any(part.isdigit() for part in url.split('.')) else 0
        url_entropy = len(set(url)) / len(url) if len(url) > 0 else 0
        
        # For location encoding, we would need to use the same encoder from training
        # For simplicity in this example, we'll use a simple hash
        location_encoded = hash(location) % 1000  # Simple encoding, in practice use the fitted encoder
        
        # Create feature array
        features = np.array([[transaction_amount, customer_age, account_balance, 
                            url_length, num_dots, has_https, num_digits, 
                            num_special_chars, has_ip, url_entropy, location_encoded]])
        
        return features
    
    def predict_single(self, url, transaction_amount, customer_age, account_balance, location):
        """Predict for a single sample"""
        # Engineer features
        features = self.engineer_features(url, transaction_amount, customer_age, account_balance, location)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return prediction, probability
    
    def predict_batch(self, df):
        """Predict for a batch of samples"""
        # Feature engineering for batch
        df_features = df.copy()
        df_features['url_length'] = df_features['url'].apply(len)
        df_features['num_dots'] = df_features['url'].apply(lambda x: x.count('.'))
        df_features['has_https'] = df_features['url'].apply(lambda x: 1 if x.startswith('https') else 0)
        df_features['num_digits'] = df_features['url'].apply(lambda x: sum(c.isdigit() for c in x))
        df_features['num_special_chars'] = df_features['url'].apply(lambda x: sum(not c.isalnum() for c in x))
        df_features['has_ip'] = df_features['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
        df_features['url_entropy'] = df_features['url'].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)
        
        # For location encoding, we would need to use the same encoder from training
        # For simplicity in this example, we'll use a simple hash
        df_features['Location_encoded'] = df_features['Location'].apply(lambda x: hash(x) % 1000)
        
        # Select features
        feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                          'url_length', 'num_dots', 'has_https', 'num_digits', 
                          'num_special_chars', 'has_ip', 'url_entropy', 'Location_encoded']
        
        X = df_features[feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

def example_usage():
    """Example of how to use the RandomForestPredictor"""
    print("Random Forest Predictor - Example Usage")
    print("=" * 40)
    
    # Initialize predictor (you would use the path to your trained model)
    # predictor = RandomForestPredictor('path/to/your/trained/model.pkl')
    
    # Example prediction for a single sample
    print("Example 1: Single Prediction")
    print("-" * 30)
    print("URL: https://secure.bankofamerica.login.verify.com")
    print("Transaction Amount: $250.50")
    print("Customer Age: 35")
    print("Account Balance: $5000.00")
    print("Location: New York")
    
    # Note: In a real implementation, you would use the actual trained model
    print("\nPrediction: PHISHING (simulated)")
    print("Confidence: 92.5%")
    
    print("\n" + "=" * 50)
    
    # Example batch prediction
    print("Example 2: Batch Prediction")
    print("-" * 30)
    sample_data = pd.DataFrame({
        'url': [
            'https://secure.bankofamerica.com',
            'https://secure.bankofamerica.login.verify.com',
            'https://paypal.com',
            'https://paypa1.com'
        ],
        'TransactionAmount': [100.0, 250.5, 75.25, 500.0],
        'CustomerAge': [30, 35, 28, 45],
        'AccountBalance': [5000.0, 7500.0, 2500.0, 10000.0],
        'Location': ['California', 'New York', 'Texas', 'Florida']
    })
    
    print("Sample data:")
    print(sample_data)
    print("\nPredictions: 2 Legitimate, 2 Phishing (simulated)")

def main():
    """Main function"""
    example_usage()
    
    print("\n" + "=" * 50)
    print("INSTRUCTIONS FOR USING THE TRAINED MODEL")
    print("=" * 50)
    print("1. First, train and save your Random Forest model using random_forest_implementation.py")
    print("2. Update the model path in RandomForestPredictor initialization")
    print("3. For accurate location encoding, save and load the fitted LabelEncoder from training")
    print("4. Use predict_single() for individual predictions or predict_batch() for multiple samples")

if __name__ == "__main__":
    main()