# 🚨 Fake Payment Gateway Detection (Adversarial‑Resistant)

A machine‑learning based system to **detect fake / phishing payment gateways** by analyzing URL metadata, transaction‑style behavior, and adversarial‑robust features.  
The project integrates **real phishing datasets**, **legitimate URLs**, **advanced preprocessing**, and **adversarial example generation** to build a strong, production‑grade classifier.

---

## ✨ Features

- 🔍 Detects fake payment gateways with ML  
- 🛡️ Adversarial‑resistant feature engineering  
- 📊 Clean, structured datasets (PhishTank + Kaggle)  
- 🤖 Models trained: SVM, XGBoost, Random Forest (comprehensive implementation)  
- 📈 High accuracy + evaluation metrics  
- 💾 Saved final models for deployment  
- 🔄 Hyperparameter tuning capabilities  
- 🎯 Prediction API for single and batch samples  

---

## 🗂️ Project Workflow

### **1️⃣ Dataset Collection**
- **PhishTank** → phishing URLs  
- **Kaggle** → legitimate/benign URLs  

### **2️⃣ Data Preprocessing**
- Cleaning  
- Removing duplicates  
- URL normalization  
- Token length checks  
- Feature extraction  

### **3️⃣ Adversarial Example Generation**
- Adding random noise  
- URL obfuscation patterns  
- Character-level mutations  
- Homoglyph injection  

### **4️⃣ Feature Engineering**
- URL-based features (length, entropy, digits count, domain age)  
- Lexical + statistical features  
- Adversarial signal features  

### **5️⃣ Model Training**  
- Based on Evaluation 
- Random Forest with hyperparameter tuning

### **6️⃣ Evaluation**
- Accuracy  
- Precision/Recall  
- F1-score  
- Confusion matrix  

### **7️⃣ Model Saving**
- Export trained models (`.pkl`)  
- Ready for production API or dashboard  

---
## 🌲 Random Forest Implementation

The project includes a simple Random Forest implementation with the following features:

- **Basic Training**: Training with optimized parameters
- **Enhanced Feature Set**: URL-based features including length, entropy, digit count, special character count
- **Model Persistence**: Save trained models for future use
- **Evaluation Metrics**: Performance evaluation including accuracy, precision, recall, F1-score, and confusion matrix
- **Feature Importance**: Analysis of which features contribute most to predictions

The implementation is located in `Data/Notebook/Models/random_forest_simple.py` and provides a streamlined interface for Random Forest functionality.



