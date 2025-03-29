import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved model

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")  # Ensure the model file is in the same directory

# Set Streamlit page configuration
st.set_page_config(page_title="Fraud Detection App", page_icon="⚠️", layout="wide")

# App title
st.title("🔍 Fraudulent Transaction Detection System")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Detect Fraud", "About"])

# Home Page
if page == "Home":
    st.write("""
    ### Welcome to the Fraud Detection App
    This application allows you to detect fraudulent transactions using a machine learning model.
    Upload a dataset or enter transaction details manually to check for fraud.
    """)

# Fraud Detection Page
elif page == "Detect Fraud":
    st.subheader("Enter Transaction Details")

    # Centering the input fields and prediction button using Streamlit columns
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column widths for alignment

    with col2:  # Centering content in column 2
        # User input fields for transaction details
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
        time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0)
        feature1 = st.number_input("Feature 1")
        feature2 = st.number_input("Feature 2")
        feature3 = st.number_input("Feature 3")
        feature4 = st.number_input("Feature 4")
        feature5 = st.number_input("Feature 5")

        # Create a dataframe for prediction
        input_data = pd.DataFrame([[time, amount, feature1, feature2, feature3, feature4, feature5]],
                                  columns=["Time", "Amount", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5"])

        if st.button("Predict Fraud"):
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Transaction is Legitimate.")

    # Option to upload CSV for batch processing
    st.subheader("Upload Transaction Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        predictions = model.predict(df)
        df["Fraud Prediction"] = ["Fraud" if pred == 1 else "Legit" for pred in predictions]
        
        # Display results in a centered layout
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write(df.head())

            # Download results button
            st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv")

# About Page
elif page == "About":
    st.write("""
    ### About This App
    - This fraud detection system uses a pre-trained machine learning model.
    - Enter transaction details manually or upload a dataset for batch processing.
    - The model predicts whether a transaction is fraudulent or legitimate.
    """)

# Footer information in the sidebar
st.sidebar.info("Developed by Your Name | AI & ML Enthusiast")
