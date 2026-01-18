# 🛒 SuperKart Sales Forecasting - MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue?logo=docker)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen)

## 📌 Business Context
**SuperKart**, a retail giant, seeks to predict future sales revenue to optimize supply chain procurement and territory planning. Accurate sales forecasting allows different verticals to chalk out their future course of action, reducing inventory risks and improving decision-making.

## 🎯 Objective
This project implements an automated **MLOps pipeline** to predict the `Product_Store_Sales_Total` (Revenue) based on historical product and store attributes. The solution leverages CI/CD practices to ensure scalability, consistency, and minimal manual intervention.

## 🏗️ Architecture & Workflow

1.  **Data Ingestion & Versioning**: Data is loaded, cleaned, and split, then versioned back to Hugging Face Datasets
2.  **Model Training & Tuning**:
    * Algorithms compared: **Random Forest**, **Gradient Boosting**, **XGBoost**
    * Hyperparameter tuning via `GridSearchCV`
    * Experiment tracking using **MLflow**
3.  **Model Evaluation**:
    * The best model is selected automatically based on the lowest **RMSE**
    * Winner: **Random Forest Regressor** (R2 Score: ~0.93, RMSE Score: ~280.85)
4.  **Deployment**:
    * The model is wrapped in a **Streamlit** web app
    * Containerized using **Docker**
    * Deployed to **Hugging Face Spaces**
5.  **CI/CD Automation**:
    * **GitHub Actions** triggers the pipeline on every push to the `main` branch

## 📂 Repository Structure

```text
├── .github/workflows/   # CI/CD Pipeline (pipeline.yml)
├── data/                # Processed datasets (optional, mostly on HF)
├── app.py               # Streamlit Application logic
├── Dockerfile           # Configuration for containerization
├── requirements.txt     # Python dependencies
├── model.joblib         # Trained Model Pipeline (Artifact)
├── README.md            # Project Documentation
└── Advanced_MLOps_SuperKart.ipynb # Research & Development Notebook
