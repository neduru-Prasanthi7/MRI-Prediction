# 🧠 MRI Prediction System using Machine Learning

An end-to-end Machine Learning web application developed to predict MRI-related outcomes using clinical and medical dataset features.

The project focuses on building a robust prediction system using data preprocessing, feature engineering, model training, and Flask deployment.

After evaluating multiple machine learning algorithms, **Gradient Boosting Classifier** was selected as the best-performing model for deployment.

---

# 📌 Project Overview

Medical datasets often contain:

- Missing values
- Outliers
- Imbalanced classes
- Complex non-linear relationships

This project addresses these challenges using preprocessing techniques and supervised machine learning to build an accurate MRI prediction system.

The trained model is deployed using Flask to provide real-time predictions through a web-based interface.

---

# 🚀 Key Features

✅ End-to-End Machine Learning Pipeline  
✅ Missing Value Handling  
✅ Outlier Detection & Variable Transformation  
✅ Feature Scaling  
✅ Data Balancing using SMOTE  
✅ Gradient Boosting Based Prediction System  
✅ ROC Curve Evaluation  
✅ Flask Web Deployment  
✅ Logging System for Model Monitoring  
✅ Real-Time Prediction Interface  

---

# 🧠 Machine Learning Workflow

## 1️⃣ Data Loading
MRI dataset loaded using Pandas.

## 2️⃣ Data Preprocessing
- Missing value handling
- Outlier treatment
- Variable transformation
- Feature scaling

## 3️⃣ Data Splitting
Dataset divided into:
- Training Data
- Testing Data

## 4️⃣ Data Balancing
SMOTE (Synthetic Minority Oversampling Technique) applied to handle class imbalance.

## 5️⃣ Model Training
The dataset was trained using Gradient Boosting Classifier for accurate prediction.

## 6️⃣ Model Evaluation
Model performance evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

## 7️⃣ Deployment
The trained model was deployed using Flask.

---

# 🏆 Best Model

## ✅ Gradient Boosting Classifier

Gradient Boosting achieved the best overall performance and was selected for deployment.

### 📊 Model Accuracy
```text
Accuracy: 76.83%
```

### Why Gradient Boosting?

- Handles non-linear relationships effectively
- Performs well on structured medical datasets
- Robust against noise and outliers
- High predictive accuracy
- Strong ensemble learning capability

---

# 📈 Model Evaluation

The model was evaluated using ROC Curve analysis and classification metrics to measure prediction performance.

### Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve
- Confusion Matrix

---

# 🌐 Flask Deployment

The trained Gradient Boosting model is integrated into a Flask web application for real-time MRI prediction.

## Backend
- Flask
- Pickle Serialization

## Frontend
- HTML
- CSS
- JavaScript

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Flask
- Matplotlib

---



# 👨‍💻 Project Objective

This project demonstrates how machine learning can be applied to medical datasets to build a real-time prediction system using Flask deployment.

The system successfully implements:
- Data preprocessing
- Feature engineering
- Machine learning prediction
- Performance evaluation
- Real-time web deployment

---

# ⭐ Conclusion

The MRI Prediction System successfully demonstrates a complete machine learning workflow from preprocessing to deployment.

Using the Gradient Boosting Classifier, the system achieved strong prediction performance and provides real-time MRI outcome prediction through a Flask web application.
