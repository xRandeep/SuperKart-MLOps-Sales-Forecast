import streamlit as st
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
HF_USERNAME = "iStillWaters" 
MODEL_REPO_NAME = "SuperKart-model"

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{MODEL_REPO_NAME}", filename="model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("🛒 SuperKart Sales Forecast App")
st.markdown("Enter product and store details to predict sales revenue.")

if model:
    with st.form("prediction_form"):
        st.subheader("Product Details")
        col1, col2 = st.columns(2)
        
        with col1:
            # Replaces Product_Id with just the category characters as requested
            product_cat_prefix = st.selectbox("Product Category (First 2 chars of ID)", ["FD", "DR", "NC"], help="FD=Food, DR=Drink, NC=Non-Consumable")
            item_weight = st.number_input("Product Weight", min_value=0.0, value=12.5)
            item_visibility = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05)
            item_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
            
        with col2:
            item_fat_content = st.selectbox("Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
            item_type = st.selectbox("Product Type", [
                'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 
                'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 
                'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 
                'Others', 'Seafood'
            ])

        st.subheader("Store Details")
        col3, col4 = st.columns(2)
        with col3:
            store_id = st.text_input("Store ID", "OUT001")
            outlet_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2026, value=1999)
            
        with col4:
            outlet_size = st.selectbox("Store Size", ['High', 'Medium', 'Small'])
            outlet_location_type = st.selectbox("City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
            outlet_type = st.selectbox("Store Type", ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'])

        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        # 1. Create Raw DataFrame (Simulating original columns)
        # We perform the Feature Engineering on this DF exactly as done in process_data.py
        input_data = pd.DataFrame({
            'Product_Weight': [item_weight],
            'Product_Sugar_Content': [item_fat_content],
            'Product_Allocated_Area': [item_visibility],
            'Product_Type': [item_type],
            'Product_MRP': [item_mrp],
            'Store_Id': [store_id],
            'Store_Establishment_Year': [outlet_year],
            'Store_Size': [outlet_size],
            'Store_Location_City_Type': [outlet_location_type],
            'Store_Type': [outlet_type],
            # We insert the prefix directly as 'Product_Category' since we asked for it directly
            # Logic: If we asked for Product_Id, we would slice it. Here we just take the input.
            'Product_Category': [product_cat_prefix] 
        })
        
        # --- FEATURE ENGINEERING (Replicating process_data.py) ---
        
        # 2a. Correct Product_Sugar_Content
        input_data['Product_Sugar_Content'] = input_data['Product_Sugar_Content'].replace(
            {'reg': 'Regular', 'low sugar': 'Low Sugar'}
        )
        
        # 2b. Product_Category 
        # (Already handled by input, but ensuring it's string)
        input_data["Product_Category"] = input_data["Product_Category"].astype(str)

        # 2c. Map Shelf Life
        perishable = [
            'Frozen Foods', 'Dairy', 'Meat', 'Fruits and Vegetables',
            'Breads', 'Breakfast', 'Seafood', 'Baking Goods', 'Starchy Foods'
        ]
        input_data["Product_Shelf_Life"] = input_data["Product_Type"].apply(
            lambda x: "Perishable" if x in perishable else "Non-Perishable"
        )

        # 2d. Create Store Age
        input_data["Store_Age"] = 2026 - input_data["Store_Establishment_Year"]

        # 2e. Store Age Bucket
        # We must use the same logic: bins range(0, max+6, 5). 
        # Assuming max age around 50 for safety (1980 to 2026 is 46 years)
        bins = range(0, 60, 5) 
        labels = [f"{b}-{b+4}" for b in bins[:-1]]
        input_data["Store_Age_Bucket"] = pd.cut(input_data["Store_Age"], bins=bins, labels=labels, right=True)
        input_data["Store_Age_Bucket"] = input_data["Store_Age_Bucket"].astype(str)

        # Drop columns that were dropped in training
        cols_to_drop = ["Product_Id", "Store_Establishment_Year", "Store_Age"]
        input_data = input_data.drop(columns=[c for c in cols_to_drop if c in input_data.columns])

        # Ensure all object columns are strings
        for col in input_data.select_dtypes(include=['object', 'category']).columns:
            input_data[col] = input_data[col].astype(str)

        # --- PREDICTION ---
        try:
            prediction = model.predict(input_data)
            st.success(f"💰 Predicted Sales Revenue: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.write("Debug Data:", input_data)