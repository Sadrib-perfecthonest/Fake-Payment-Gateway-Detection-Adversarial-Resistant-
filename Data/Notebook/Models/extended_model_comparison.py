import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data_extended():
    """Load and prepare the feature data for extended model training"""
    # Load the combined dataset
    df = pd.read_csv('../../Processed/combined_dataset.csv')
    
    # Enhanced feature engineering
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
    
    # Select features for modeling
    feature_columns = ['TransactionAmount', 'CustomerAge', 'AccountBalance', 
                      'url_length', 'num_dots', 'has_https', 'num_digits', 
                      'num_special_chars', 'has_ip', 'url_entropy', 'Location_encoded']
    
    X = df[feature_columns]
    y = df['label']
    
    # Handle missing values
    X = X.fillna(0)
    
    return X, y, df

def train_and_evaluate_models_extended(X, y):
    """Train multiple models and evaluate their performance"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale the features (needed for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to evaluate
    models = {
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=1000), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10), False),
        'Decision Tree': (DecisionTreeClassifier(random_state=42, max_depth=10), False),
        'SVM': (SVC(random_state=42, probability=True), True),
        'XGBoost': (xgb.XGBClassifier(random_state=42), False),
        'AdaBoost': (AdaBoostClassifier(random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=42), False),
        'Naive Bayes': (GaussianNB(), False),
        'K-Nearest Neighbors': (KNeighborsClassifier(), True),
        'Neural Network': (MLPClassifier(random_state=42, max_iter=1000), True)
    }
    
    # Store results
    results = {}
    
    print("\n" + "="*70)
    print("EXTENDED MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Train and evaluate each model
    for name, (model, needs_scaling) in models.items():
        print(f"\n{name}:")
        print("-" * (len(name) + 1))
        
        # Use scaled data if needed
        if needs_scaling:
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
    
    return results

def main():
    """Main function to run the extended model comparison"""
    print("Fake Payment Gateway Detection - Extended Model Comparison")
    print("=" * 55)
    
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data_extended()
        print(f"Dataset shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Train and evaluate models
        results = train_and_evaluate_models_extended(X, y)
        
        # Find best models based on F1 score
        print("\n" + "="*70)
        print("BEST MODEL COMPARISON (based on F1-Score)")
        print("="*70)
        
        # Sort models by F1 score
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        print(f"\nTop 5 Models by F1-Score:")
        print("-" * 40)
        for i, (name, metrics) in enumerate(sorted_models[:5]):
            f1 = metrics['f1_score']
            marker = " ← Best" if i == 0 else ""
            print(f"{i+1}. {name:25s}: {f1:.4f}{marker}")
            
        # Summary table for all models
        print("\nSummary of All F1-Scores:")
        print("-" * 45)
        for name, metrics in sorted_models:
            f1 = metrics['f1_score']
            print(f"{name:25s}: {f1:.4f}")
            
        # Highlight Random Forest performance
        rf_f1 = results['Random Forest']['f1_score']
        print(f"\nRandom Forest F1-Score: {rf_f1:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()