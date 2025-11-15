# 🚨 Fake Payment Gateway Detection (Adversarial‑Resistant)

A machine‑learning based system to **detect fake / phishing payment gateways** by analyzing URL metadata, transaction‑style behavior, and adversarial‑robust features.  
The project integrates **real phishing datasets**, **legitimate URLs**, **advanced preprocessing**, and **adversarial example generation** to build a strong, production‑grade classifier.

---

## ✨ Features

- 🔍 Detects fake payment gateways with ML  
- 🛡️ Adversarial‑resistant feature engineering  
- 📊 Clean, structured datasets (PhishTank + Kaggle)  
- 🤖 Models trained: SVM, XGBoost (extendable)  
- 📈 High accuracy + evaluation metrics  
- 💾 Saved final models for deployment  

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

### **6️⃣ Evaluation**
- Accuracy  
- Precision/Recall  
- F1-score  
- Confusion matrix  

### **7️⃣ Model Saving**
- Export trained models (`.pkl`)  
- Ready for production API or dashboard  

---

## 📁 Folder Structure

