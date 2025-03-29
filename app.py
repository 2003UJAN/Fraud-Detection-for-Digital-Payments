import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time

# Load Model
model = keras.models.load_model("fraud_detection_model.h5")

# Streamlit Page Config
st.set_page_config(page_title="Fraud Detection AI", layout="wide")

# Custom CSS for 3D UI
def custom_css():
    st.markdown(
        """
        <style>
            body {background-color: #121212; color: white;}
            .sidebar .sidebar-content {background-color: #1e1e1e;}
            .stButton>button {border-radius: 12px; box-shadow: 2px 2px 10px #888888;}
            .stTextInput>div>div>input {border-radius: 10px; padding: 10px;}
            .stNumberInput>div>div>input {border-radius: 10px; padding: 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

custom_css()

# Sidebar - Transaction Inputs
st.sidebar.header("🔍 Enter Transaction Details")
amount = st.sidebar.number_input("💰 Transaction Amount ($)", min_value=1, max_value=10000, value=100)
time_of_transaction = st.sidebar.slider("⏳ Transaction Hour (0-23)", 0, 23, 12)
location = st.sidebar.selectbox("📍 Location", ["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])
device = st.sidebar.selectbox("📱 Device Type", ["Mobile", "Desktop", "Tablet", "Smartwatch"])
transaction_type = st.sidebar.selectbox("💳 Transaction Type", ["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])
frequency = st.sidebar.slider("🔁 Transactions per Month", 1, 50, 5)
past_fraud = st.sidebar.radio("⚠️ Past Fraud History", ["No", "Yes"])

# Encoding Inputs
location_map = {loc: i for i, loc in enumerate(["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])}
device_map = {dev: i for i, dev in enumerate(["Mobile", "Desktop", "Tablet", "Smartwatch"])}
transaction_map = {t: i for i, t in enumerate(["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])}

# Prepare input for model
input_data = np.array([[
    amount, 
    time_of_transaction, 
    location_map[location], 
    device_map[device], 
    transaction_map[transaction_type], 
    frequency, 
    1 if past_fraud == "Yes" else 0
]])

scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# Fraud Detection
if st.sidebar.button("🔎 Detect Fraud"):
    with st.spinner("Analyzing Transaction..."):
        time.sleep(2)
        prediction = model.predict(input_scaled)
        fraud_prob = float(prediction[0][0]) * 100  # Convert to percentage
        
    st.sidebar.subheader(f"📊 Fraud Risk Score: {fraud_prob:.2f}%")
    
    if fraud_prob > 50:
        st.sidebar.error("🚨 High Risk! Fraud Detected.")
    else:
        st.sidebar.success("✅ Transaction is Safe.")

# 3D Styled Fraud Detection Dashboard
st.markdown("""
    <h1 style='text-align: center;'>🚀 AI Fraud Detection System</h1>
    <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/5477/5477844.png' width=150>
    </div>
    <p style='text-align: center; font-size: 18px;'>
        Secure your transactions with AI-powered fraud detection.
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center;'>
        <button style='border-radius: 12px; padding: 10px 20px; font-size: 18px; background-color: #ff4b4b; color: white; border: none;'>
            AI-Powered Security ✅
        </button>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("**Developed with Streamlit & TensorFlow | AI-Powered Fraud Detection 🔒**")
