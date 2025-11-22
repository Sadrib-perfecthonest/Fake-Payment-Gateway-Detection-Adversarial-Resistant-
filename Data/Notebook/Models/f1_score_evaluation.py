import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the feature data for model training"""
    # Load the feature data
    df = pd.read_csv('../../../Data/Processed/Feature.csv')
    
    # Display basic info about the dataset
    print("Dataset shape:", df.shape)
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # Prepare features and target
    # Since the Feature.csv doesn't seem to have engineered features, we'll create some basic ones
    df['url_length'] = df['url'].apply(len)
    df['num_dots'] = df['url'].apply(lambda x: x.count('.'))
    df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https') else 0)
    df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Location_encoded'] = le.fit_transform(df['Location'])
    
    # Select features for modeling
    feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                      'url_length', 'num_dots', 'has_https', 'num_digits', 'Location_encoded']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, df

def train_and_evaluate_models(X, y):
    """Train multiple models and evaluate their performance"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    # Store results
    results = {}
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * (len(name) + 1))
        
        # For SVM, use scaled data; for others, use original data
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'model': model
        }
        
        # Display metrics
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print("\n  Classification Report:")
        print("  " + "-" * 25)
        report = classification_report(y_test, y_pred)
        print("  " + str(report).replace('\n', '\n  '))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"  {cm}")
    
    return results, X_test, y_test

def main():
    """Main function to run the evaluation"""
    print("Fake Payment Gateway Detection - Model Evaluation")
    print("=" * 50)
    
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data()
        
        # Train and evaluate models
        results, X_test, y_test = train_and_evaluate_models(X, y)
        
        # Find best model based on F1 score
        print("\n" + "="*60)
        print("BEST MODEL COMPARISON (based on F1-Score)")
        print("="*60)
        
        best_f1 = 0
        best_model_name = ""
        
        for name, metrics in results.items():
            f1 = metrics['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {best_f1:.4f})")
        
        # Summary table
        print("\nSummary of F1-Scores:")
        print("-" * 30)
        for name, metrics in results.items():
            f1 = metrics['f1_score']
            marker = " ← Best" if name == best_model_name else ""
            print(f"{name:20s}: {f1:.4f}{marker}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()