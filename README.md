# 🧠 Fraud Detection Using Machine Learning

## 📌 Project Overview
This project develops a machine learning-based application to classify financial transactions as either **fraudulent or legitimate** using four supervised learning algorithms:  
- Artificial Neural Network (ANN)  
- Support Vector Machine (SVM)  
- Decision Tree (DT)  
- k-Nearest Neighbors (K-NN)  

The project follows a full ML pipeline, from preprocessing and feature selection to model evaluation and result export. Built as part of the CMPS 470 Final Project by **Code Crafters**.

---

## 👨‍💻 Team Contributions
**Fiyinfoluwa Osifala** — Model implementation, visualizations, GitHub repo management  
**Shakurah Watson** — Data preparation and testing

---

## 🧰 Tools & Technologies
- Python, scikit-learn, Keras
- Jupyter Notebook, Excel
- GitHub, CLI interface (command-line)

---

## 🧪 Dataset & Features
- Simulated financial transactions dataset  
- **Features used**: `Amount` and `Account_age`  
- **Target**: `Fraud_Label` (0 = Not Fraud, 1 = Fraud)  
- The dataset is **imbalanced** — most transactions are legitimate

---

## 🔄 ML Pipeline Summary

| Step               | Description |
|--------------------|-------------|
| **1. Data Cleaning** | Removed duplicates, ensured no missing values |
| **2. Preprocessing** | StandardScaler used only for ANN & SVM |
| **3. Feature Selection** | Used `Amount` and `Account_age` (continuous, numeric) |
| **4. Modeling** | Implemented ANN, SVM, K-NN, DT |
| **5. Evaluation** | Accuracy, Precision, Recall, F1-score, AUC, Confusion Matrix |
| **6. Output** | Saved trained models and predictions to organized folders |

> 📁 Folder structure includes: `INPUT/`, `OUTPUT/`, `MODEL/`, `DOC/`, `CODE/`, and `OTHER/`

---

## 📈 Model Performance

| Model     | Accuracy | Fraud Precision | Fraud Recall | Notes |
|-----------|----------|-----------------|---------------|-------|
| **ANN**   | 87%      | Good             | Balanced       | Best overall performer |
| **SVM**   | ~85%     | Perfect          | Low            | Missed some true frauds |
| **K-NN**  | ~83%     | Perfect          | Low            | Similar to SVM |
| **DT**    | ~80%     | 50%              | 50%            | Modest balance |

> *Due to class imbalance, most models favored non-fraud predictions.*

---

## ✅ Key Insights & Challenges
- ANN had the most balanced performance across all metrics  
- SVM and K-NN were **high precision but low recall** (missed many frauds)  
- Decision Trees were easy to interpret but gave moderate results  
- Class imbalance heavily influenced results — future versions could apply **resampling** or **threshold tuning**

---

## 🗂 Project Structure
