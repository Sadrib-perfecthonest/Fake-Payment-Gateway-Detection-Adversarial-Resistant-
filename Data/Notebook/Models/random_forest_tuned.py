import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the feature data for Random Forest model training"""
    # Load the combined dataset
    df = pd.read_csv('D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Processed/combined_dataset.csv')
    
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
    
    return X, y, df, feature_columns

def tune_random_forest(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    print("Performing hyperparameter tuning for Random Forest...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_and_evaluate_rf(X, y):
    """Train and evaluate Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Tune Random Forest model
    best_rf = tune_random_forest(X_train, y_train)
    
    # Make predictions with tuned model
    y_pred = best_rf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return best_rf, X_test, y_test, y_pred, accuracy, precision, recall, f1

def display_feature_importance(model, feature_names):
    """Display feature importance from the Random Forest model"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance Rankings:")
    print("-" * 40)
    for i in range(min(10, len(feature_names))):  # Show top 10 features
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    """Main function to run the tuned Random Forest implementation"""
    print("Fake Payment Gateway Detection - Tuned Random Forest Implementation")
    print("=" * 70)
    
    try:
        # Load and prepare data
        X, y, df, feature_columns = load_and_prepare_data()
        print(f"Dataset shape: {df.shape}")
        print("Label distribution:")
        print(df['label'].value_counts())
        
        # Train and evaluate Random Forest model
        model, X_test, y_test, y_pred, accuracy, precision, recall, f1 = train_and_evaluate_rf(X, y)
        
        # Display results
        print("\n" + "="*50)
        print("TUNED RANDOM FOREST MODEL PERFORMANCE")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Display classification report
        print("\nDetailed Classification Report:")
        print("-" * 40)
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Display feature importance
        display_feature_importance(model, feature_columns)
        
        # Save the trained model
        import joblib
        joblib.dump(model, 'D:/python project(cse466)/Fake-Payment-Gateway-Detection-Adversarial-Resistant-/Data/Notebook/Models/tuned_random_forest_model.pkl')
        print("\nTuned model saved successfully as 'tuned_random_forest_model.pkl'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()