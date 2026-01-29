# 📞 Telecom Customer Churn Prediction using Machine Learning

---

## 🔍 Project Overview
Customer churn is one of the most critical problems faced by telecom companies. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project focuses on **predicting customer churn using machine learning** and delivering **actionable insights through a live dashboard and prediction application**.

The project includes:
- End-to-end **Machine Learning pipeline**
- **Exploratory Data Analysis (EDA)**
- Multiple ML model training and comparison
- **Live prediction web app**
- **Live analytics dashboard**

---

## 🚀 Live Deployment
The project is deployed using **Streamlit Cloud**, providing both prediction and analytics capabilities.

### 🔗 Live Links
- **Prediction App**  
  👉 https://telecom-customers-churn-ml.streamlit.app/

- **Analytics Dashboard**  
  👉 https://telecom-customers-churn-dashboard.streamlit.app/

---

## 🎯 Objectives
- Analyze customer behavior and service usage
- Identify key drivers responsible for churn
- Build and compare multiple ML models
- Predict churn probability for new customers
- Provide business-ready insights via dashboards

---

## 💼 Business Problem & Impact
Telecom companies lose significant revenue due to customer churn.  
This project helps businesses:

- Identify **high-risk churn customers**
- Take **proactive retention actions**
- Reduce customer acquisition cost
- Improve customer lifetime value (CLV)

This solution can directly support **marketing, retention, and customer success teams**.

---

## 🔄 End-to-End ML Pipeline
The project follows a **production-oriented ML workflow**:

1. Data ingestion & understanding  
2. Data cleaning & preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature engineering  
5. Model training & comparison  
6. Model evaluation  
7. Best model selection  
8. Model persistence (`.pkl`)  
9. Deployment using Streamlit  
10. Dashboard development for insights  

---

## 🧠 Machine Learning Models Used
The following algorithms were implemented and evaluated:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier

📌 **Random Forest** was selected for deployment due to:
- Higher accuracy
- Better generalization
- Robust handling of non-linear relationships

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to:
- Understand customer demographics
- Analyze service usage patterns
- Identify churn-related trends
- Handle missing values and outliers
- Encode categorical variables
- Scale numerical features

Visualizations include:
- Churn distribution
- Contract type vs churn
- Monthly charges vs churn
- Tenure analysis
- Correlation heatmaps

---

## 🧩 Feature Engineering & Preprocessing
Key preprocessing steps:
- Encoding categorical features
- Scaling numerical variables using `StandardScaler`
- Feature consistency enforcement during inference
- Saving preprocessing objects (`scaler.pkl`, `feature_columns.pkl`)

This ensures **training and prediction pipelines remain identical**.

---

## 🧪 Model Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 🖥️ Streamlit Prediction App Features
- Clean and intuitive UI
- Takes customer details as input
- Predicts churn (Yes / No)
- Real-time ML inference
- Designed for non-technical business users

---

## 📈 Streamlit Dashboard Features
- KPI cards (Churn Rate, Total Customers, Active Customers)
- Interactive and dynamic visualizations
- Business-focused insights
- Responsive and clean layout

---

## 🛠️ Technologies Used
- **Language**: Python  
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit
- **Deployment**: Streamlit Cloud  
- **Version Control**: Git & GitHub  

---

## 📂 Project Folder Structure

```text
Telecom-Customers-Churn-ML/
│
├── app/
│   └── Streamlit app components and UI logic
│
├── assets/
│   └── Images and static assets
│
├── dashboard/
│   └── Dashboard modules and visualization logic
│
├── data/
│   └── Raw and processed datasets
│
├── jupyter files/
│   └── EDA and model training notebooks
│
├── model/
│   └── Saved ML model artifacts (pkl files)
│
├── README.md                   # Project documentation
├── app.py                      # Main Streamlit prediction app
├── churn_model.pkl             # Trained ML model
├── dashboard.py                # Streamlit dashboard script
├── feature_columns.pkl         # Model feature columns
├── scaler.pkl                  # Feature scaling object
├── requirements.txt            # Python dependencies
├── telecom.png                 # Project/banner image
└── telecom_dashboard.csv       # Dashboard dataset

```
---
## How to Run the Project Locally

Follow the steps below to run the project on your local machine:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/pavan-ahire/Telecom-Customers-Churn-ML.git
cd Telecom-Customers-Churn-ML
```
### Install Required Dependencies
- pip install -r requirements.txt
  
### Run streamlit prediction app
- streamlit run app.py
  
### Run Streamlit Dashboard
-streamlit run dashboard.py

---

## 🧠 Key Skills Demonstrated

- Machine Learning model development and evaluation
- Exploratory Data Analysis (EDA)
- Feature engineering and data preprocessing
- Model serialization and reuse (`.pkl` files)
- Deployment of ML models using Streamlit
- Dashboard creation for business insights
- End-to-end project implementation
- Version control using Git & GitHub
---

## 👨‍💻 Author

**Pavan Ahire**


 Aspiring Data Scientist | Machine Learning & Analytics Enthusiast
- [🔗 GitHub](https://github.com/pavan-ahire)
- [🔗 LinkedIn](https://www.linkedin.com/in/pavan-ahire-260940364/)


