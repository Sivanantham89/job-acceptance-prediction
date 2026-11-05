# 🎯 Job Acceptance Prediction System

## 📌 Project Overview
The Job Acceptance Prediction System is a Machine Learning application designed to predict whether a candidate is likely to get placed based on academic performance, technical skills, aptitude, communication skills, and career-related information.

The project uses a trained Machine Learning model and a Streamlit web application to provide predictions through an interactive user interface.

## 🚀 Features
- Candidate information input form
- Job placement prediction
- Placement score calculation
- Prediction history tracking
- Analytics dashboard
- CSV download functionality
- Interactive Streamlit interface

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Machine Learning

## 📂 Project Structure
```text
job_acceptance_prediction/
│
├── models/
│   └── final_model.pkl
│
├── python/
│   ├── app.py
│   └── predict.py
│
├── reports/
│   └── prediction_history/
│
├── README.md
└── requirements.txt
```

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd job_acceptance_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run python/app.py
```

## 📊 Prediction Classes
- PLACED
- NOT PLACED

## 🎓 Project Objective
To demonstrate how Machine Learning can be used to support job placement prediction using candidate-related features.

## ⚠️ Disclaimer
This application provides predictions based on a trained model. It does not guarantee employment outcomes.
