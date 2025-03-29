import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load the trained fraud detection model
model = keras.models.load_model("fraud_detection_model.h5")

# Streamlit Page Configuration
st.set_page_config(page_title="Fraud Detection AI", layout="centered")

# Sidebar for User Input
st.sidebar.header("🔍 Enter Transaction Details")
amount = st.sidebar.number_input("💰 Transaction Amount ($)", min_value=1, max_value=10000, value=100)
time = st.sidebar.slider("🕒 Transaction Hour (0-23)", 0, 23, 12)
location = st.sidebar.selectbox("📍 Transaction Location", ["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])
device = st.sidebar.selectbox("📱 Device Type", ["Mobile", "Desktop", "Tablet", "Smartwatch"])
transaction_type = st.sidebar.selectbox("💳 Transaction Type", ["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])
frequency = st.sidebar.slider("🔁 Transactions per Month", 1, 50, 5)
past_fraud = st.sidebar.radio("⚠️ Past Fraud History", ["No", "Yes"])

# Mapping categorical values
location_map = {loc: i for i, loc in enumerate(["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])}
device_map = {dev: i for i, dev in enumerate(["Mobile", "Desktop", "Tablet", "Smartwatch"])}
transaction_map = {t: i for i, t in enumerate(["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])}

# Prepare input data
input_data = np.array([[amount, time, location_map[location], device_map[device], transaction_map[transaction_type], frequency, 1 if past_fraud == "Yes" else 0]])
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# Centered UI for Prediction
st.markdown("## 🛡️ AI-Powered Fraud Detection")
st.markdown("### Secure Your Transactions with AI")

# Predict Fraud
if st.button("🔎 Detect Fraud", use_container_width=True):
    prediction = model.predict(input_scaled)
    fraud_prob = float(prediction[0][0]) * 100  # Convert to percentage
    
    if fraud_prob > 50:
        st.error(f"🚨 High Risk! Fraud Probability: {fraud_prob:.2f}%", icon="⚠️")
    else:
        st.success(f"✅ Transaction is Safe! Fraud Probability: {fraud_prob:.2f}%", icon="✔️")

# Footer
st.markdown("---")
st.markdown
