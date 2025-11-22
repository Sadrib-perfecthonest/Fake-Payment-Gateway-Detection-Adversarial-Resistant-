import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the feature data for Random Forest model training"""
    # Load the combined dataset
    df = pd.read_csv('D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/combined_dataset.csv')
    
    # Feature engineering
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_special_chars'] = df['url'].apply(lambda x: sum(not c.isalnum() for c in x))
    df['has_ip'] = df['url'].apply(lambda x: 1 if any(part.isdigit() for part in x.split('.')) else 0)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Location_encoded'] = le.fit_transform(df['Location'])
    
    # Select features for modeling
    feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                      'url_length', 'num_dots', 'has_https', 'num_digits', 
                      'num_special_chars', 'has_ip', 'Location_encoded']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, df, feature_columns

def train_random_forest_model(X, y):
    """Train a Random Forest model with optimized parameters"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Create Random Forest model with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all processors
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return rf_model, X_test, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1

def display_feature_importance(model, feature_names):
    """Display feature importance from the Random Forest model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance Rankings:")
    print("-" * 40)
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def evaluate_model_performance(accuracy, precision, recall, f1):
    """Display model performance metrics"""
    print("\n" + "="*50)
    print("RANDOM FOREST MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

def main():
    """Main function to run the Random Forest implementation"""
    print("Fake Payment Gateway Detection - Random Forest Implementation")
    print("=" * 65)
    
    try:
        # Load and prepare data
        X, y, df, feature_columns = load_and_prepare_data()
        print(f"Dataset shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Train Random Forest model
        model, X_test, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1 = train_random_forest_model(X, y)
        
        # Evaluate model performance
        evaluate_model_performance(accuracy, precision, recall, f1)
        
        # Display classification report
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Display feature importance
        display_feature_importance(model, feature_columns)
        

        # Save the trained model
        import joblib
        joblib.dump(model, 'D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Notebook/Models/random_forest_model.pkl')
        print("\nModel saved successfully as 'random_forest_model.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()