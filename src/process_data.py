import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from huggingface_hub import login

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "SuperKart-data")
HF_TOKEN = os.getenv("HF_TOKEN")

def process_data():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
    print(f"Loading raw data from {repo_id}...")
    
    try:
        # Load from the 'raw' config
        dataset = load_dataset(repo_id, "raw")
        df = dataset['train'].to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Starting Feature Engineering...")
    
    # ---------------------------------------------------------
    # 2a. Correct the Product_Sugar_Content values
    # ---------------------------------------------------------
    # Replace 'reg' with 'Regular' and 'low sugar' with 'Low Sugar'
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace(
            {'reg': 'Regular', 'low sugar': 'Low Sugar'}
        )

    # ---------------------------------------------------------
    # 2b. Create Product categories based on Product ID
    # ---------------------------------------------------------
    # Extract first 2 chars of Product_Id to separate column Product_Category
    if 'Product_Id' in df.columns:
        df["Product_Category"] = df["Product_Id"].astype(str).str[:2]
    else:
        print("Warning: 'Product_Id' missing, skipping Product_Category creation.")

    # ---------------------------------------------------------
    # 2c. Map Food categories to primary categories
    # ---------------------------------------------------------
    perishable = [
        'Frozen Foods', 'Dairy', 'Meat', 'Fruits and Vegetables',
        'Breads', 'Breakfast', 'Seafood', 'Baking Goods', 'Starchy Foods'
    ]
    
    # Create Product_Shelf_Life column
    if 'Product_Type' in df.columns:
        df["Product_Shelf_Life"] = df["Product_Type"].apply(
            lambda x: "Perishable" if x in perishable else "Non-Perishable"
        )

    # ---------------------------------------------------------
    # 2d & 2e. Create Store Age and Bucket it
    # ---------------------------------------------------------
    if 'Store_Establishment_Year' in df.columns:
        # Assuming current data provided is for current year (2026)
        df["Store_Age"] = 2026 - df["Store_Establishment_Year"]
        
        # Bucket into 5-year intervals
        # Create bins up to max age + 6 to ensure coverage
        max_age = df["Store_Age"].max()
        bins = range(0, int(max_age) + 6, 5) 
        labels = [f"{b}-{b+4}" for b in bins[:-1]]
        
        df["Store_Age_Bucket"] = pd.cut(df["Store_Age"], bins=bins, labels=labels, right=True)
        
        # Convert the categorical bucket to string for safe serialization
        df["Store_Age_Bucket"] = df["Store_Age_Bucket"].astype(str)

    # ---------------------------------------------------------
    # Drop unnecessary columns
    # ---------------------------------------------------------
    # Removing Product_Id (high cardinality), Store_Establishment_Year (replaced by bucket), Store_Age (temp)
    cols_to_drop = ["Product_Id", "Store_Establishment_Year", "Store_Age"]
    print(f"Dropping columns: {cols_to_drop}")
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Ensure all object columns are strings (avoids PyArrow/HF dataset issues)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].astype(str)

    # ---------------------------------------------------------
    # Split and Upload
    # ---------------------------------------------------------
    print("Splitting data...")
    if 'Product_Store_Sales_Total' not in df.columns:
        print("Error: Target variable 'Product_Store_Sales_Total' missing.")
        return

    X = df.drop(columns=['Product_Store_Sales_Total'])
    y = df['Product_Store_Sales_Total']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Recombine for HF Dataset format
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    processed_dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df)
    })
    
    print(f"Uploading processed data to {repo_id}...")
    processed_dataset.push_to_hub(repo_id, config_name="processed", token=HF_TOKEN)
    print("Processed data uploaded successfully.")

if __name__ == "__main__":
    process_data()