import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# --- CONFIGURATION ---
HF_USERNAME = os.getenv("HF_USERNAME", "iStillWaters")
DATASET_REPO_NAME = os.getenv("DATASET_REPO_NAME", "SuperKart-data")
HF_TOKEN = os.getenv("HF_TOKEN")

# Path to your local raw data
RAW_DATA_PATH = "data/SuperKart.csv" 

def register_raw_data():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print("Error: SuperKart.csv not found in 'data/' folder.")
        return

    # Convert to HF Dataset
    # We use a DatasetDict to organize it, though it's just one split 'train' for now
    dataset = DatasetDict({
        "train": Dataset.from_pandas(df)
    })

    # Push to Hub
    repo_id = f"{HF_USERNAME}/{DATASET_REPO_NAME}"
    print(f"Uploading raw data to {repo_id}...")
    
    dataset.push_to_hub(repo_id, config_name="raw", token=HF_TOKEN)
    print("Raw data registered successfully.")

if __name__ == "__main__":
    register_raw_data()