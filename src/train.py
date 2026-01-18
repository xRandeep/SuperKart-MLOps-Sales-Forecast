import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from huggingface_hub import HfApi, login

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "SuperKart-data")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME", "SuperKart-model")
HF_TOKEN = os.getenv("HF_TOKEN")

def train_and_evaluate():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    # 1. Load Processed Data
    data_repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
    print(f"Loading processed data from {data_repo_id}...")
    dataset = load_dataset(data_repo_id, "processed")
    
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Clean artifacts (HF specific)
    if '__index_level_0__' in train_df.columns:
        train_df = train_df.drop(columns=['__index_level_0__'])
        test_df = test_df.drop(columns=['__index_level_0__'])

    X_train = train_df.drop('Product_Store_Sales_Total', axis=1)
    y_train = train_df['Product_Store_Sales_Total']
    X_test = test_df.drop('Product_Store_Sales_Total', axis=1)
    y_test = test_df['Product_Store_Sales_Total']

    # 2. Define Preprocessing Pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # 3. Define Models & Params
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    param_grids = {
        'RandomForest': {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [5, 10]},
        'GradientBoosting': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.1]},
        'XGBoost': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.1]}
    }

    # 4. Training Loop
    best_overall_rmse = float('inf')
    best_overall_model = None
    best_model_name = ""
    
    # Set experiment
    mlflow.set_experiment("SuperKart_Sales_Forecast")

    print("Starting Training & Tuning...")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
            
            grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, 
                                       scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            
            print(f"Model: {model_name} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
            
            if rmse < best_overall_rmse:
                best_overall_rmse = rmse
                best_overall_model = best_model
                best_model_name = model_name

    print(f"\nWinner: {best_model_name} (RMSE: {best_overall_rmse:.4f})")

    # 5. Save & Register Model
    joblib.dump(best_overall_model, "model.joblib")
    
    # Create Readme for Model Card
    model_card = f"""
---
tags:
- sklearn
- regression
- sales-forecast
metrics:
- rmse
model-index:
- name: {best_model_name}
  results:
  - task:
      type: tabular-regression
    metrics:
      - type: rmse
        value: {best_overall_rmse}
---
# SuperKart Sales Prediction Model
This is a **{best_model_name}** model trained on the SuperKart dataset.
    """
    with open("README.md", "w") as f:
        f.write(model_card)

    # Upload to HF Model Hub
    api = HfApi()
    model_repo_id = f"{HF_USERNAME}/{MODEL_REPO_NAME}"
    
    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, token=HF_TOKEN)
    
    api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", 
                    repo_id=model_repo_id, repo_type="model", token=HF_TOKEN)
    api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", 
                    repo_id=model_repo_id, repo_type="model", token=HF_TOKEN)
    
    print(f"Model registered at https://huggingface.co/{model_repo_id}")

if __name__ == "__main__":
    train_and_evaluate()