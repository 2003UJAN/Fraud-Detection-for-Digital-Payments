import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load Model
model = keras.models.load_model("fraud_detection_model.h5")

# Configure UI
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #121212; color: white; }
        .stButton button { border-radius: 12px; font-size: 18px; padding: 10px; }
        .stSidebar { background-color: #1e1e1e; padding: 20px; border-radius: 15px; }
        .stTitle { text-align: center; font-size: 36px; font-weight: bold; color: #ff6600; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Enter Transaction Details")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1, max_value=10000, value=100)
time = st.sidebar.slider("Transaction Hour (0-23)", 0, 23, 12)
location = st.sidebar.selectbox("Transaction Location", ["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])
device = st.sidebar.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Smartwatch"])
transaction_type = st.sidebar.selectbox("Transaction Type", ["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])
frequency = st.sidebar.slider("Transactions per Month", 1, 50, 5)
past_fraud = st.sidebar.radio("Past Fraud History", [0, 1])

# Mapping for categorical features
location_map = {loc: i for i, loc in enumerate(["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])}
device_map = {dev: i for i, dev in enumerate(["Mobile", "Desktop", "Tablet", "Smartwatch"])}
transaction_map = {t: i for i, t in enumerate(["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])}

# Process input data
input_data = np.array([[amount, time, location_map[location], device_map[device], transaction_map[transaction_type], frequency, past_fraud]])
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# Fraud Detection
if st.sidebar.button("🔎 Detect Fraud"):
    prediction = model.predict(input_scaled)
    fraud_prob = float(prediction[0][0])  # Probability of fraud

    st.sidebar.subheader("📊 Fraud Risk Score: {:.2f}%".format(fraud_prob * 100))

    if fraud_prob > 0.5:
        st.sidebar.error("🚨 High Risk! Fraud Detected.")
    else:
        st.sidebar.success("✅ Transaction is Safe.")

# UI Enhancements
st.markdown('<p class="stTitle">📊 AI-Powered Fraud Detection</p>', unsafe_allow_html=True)
st.markdown("🚀 **Powered by Streamlit & TensorFlow** | Secure Your Transactions")
