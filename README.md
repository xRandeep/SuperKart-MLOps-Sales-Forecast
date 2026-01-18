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
```

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/SuperKart-MLOps-Sales-Forecast.git](https://github.com/YOUR_USERNAME/SuperKart-MLOps-Sales-Forecast.git)
cd SuperKart-MLOps-Sales-Forecast
```
**2. Install Dependencies
```Bash
pip install -r requirements.txt
```
**3. Run the App
```Bash
streamlit run app.py
```

## 📊 Model Performance

After extensive hyperparameter tuning and cross-validation, the **Random Forest** model outperformed others.

| Model | RMSE | R2 Score |
| :--- | :--- | :--- |
| **Random Forest** | **280.85** | **0.93** |
| Gradient Boosting | 284.43 | 0.92 |
| XGBoost | 285.58 | 0.92 |

## 🔗 Deployment

The application is live and can be accessed here:
**[Link to your Hugging Face Space]**

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
