import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

model = keras.models.load_model("fraud_detection_model.h5")

df = pd.read_csv("advanced_fraud_transactions.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["fraudulent"]))

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.sidebar.header("🔍 Enter Transaction Details")

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1, max_value=10000, value=100)
time = st.sidebar.slider("Transaction Hour (0-23)", 0, 23, 12)
location = st.sidebar.selectbox("Transaction Location", ["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])
device = st.sidebar.selectbox("Device Type", ["Mobile", "Desktop", "Tablet", "Smartwatch"])
transaction_type = st.sidebar.selectbox("Transaction Type", ["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])
frequency = st.sidebar.slider("Transactions per Month", 1, 50, 5)
past_fraud = st.sidebar.radio("Past Fraud History", [0, 1])

location_map = {loc: i for i, loc in enumerate(["NYC", "SFO", "LON", "SGP", "TKY", "SYD", "BER", "PAR"])}
device_map = {dev: i for i, dev in enumerate(["Mobile", "Desktop", "Tablet", "Smartwatch"])}
transaction_map = {t: i for i, t in enumerate(["Online Purchase", "ATM Withdrawal", "POS Payment", "Bank Transfer", "Crypto Exchange"])}

input_data = np.array([[amount, time, location_map[location], device_map[device], transaction_map[transaction_type], frequency, past_fraud]])
input_scaled = scaler.transform(input_data)

if st.sidebar.button("🔎 Detect Fraud"):
    prediction = model.predict(input_scaled)
    fraud_prob = float(prediction[0][0])  # Probability of fraud

    st.sidebar.subheader("📊 Fraud Risk Score: {:.2f}%".format(fraud_prob * 100))

    if fraud_prob > 0.5:
        st.sidebar.error("🚨 High Risk! Fraud Detected.")
    else:
        st.sidebar.success("✅ Transaction is Safe.")

st.title("📊 Fraud Detection Dashboard")

st.subheader("Fraud vs Legitimate Transactions")
fig, ax = plt.subplots()
sns.countplot(x=df["fraudulent"], palette=["green", "red"], ax=ax)
ax.set_xticklabels(["Legit", "Fraud"])
st.pyplot(fig)

st.subheader("Transaction Amount Distribution")
fig, ax = plt.subplots()
sns.histplot(df[df["fraudulent"] == 0]["transaction_amount"], color="green", label="Legit", kde=True, ax=ax)
sns.histplot(df[df["fraudulent"] == 1]["transaction_amount"], color="red", label="Fraud", kde=True, ax=ax)
ax.legend()
st.pyplot(fig)

st.subheader("Fraud Transactions by Hour")
fig, ax = plt.subplots()
sns.histplot(df[df["fraudulent"] == 1]["transaction_time"], bins=24, color="red", kde=True, ax=ax)
ax.set_xlabel("Hour of the Day")
st.pyplot(fig)

st.subheader("Fraud by Device Type")
fig, ax = plt.subplots()
sns.barplot(x=df["device_type"], y=df["fraudulent"], palette="coolwarm", ax=ax)
st.pyplot(fig)

st.markdown("🚀 **Developed with Streamlit & TensorFlow** | AI-Powered Fraud Detection")
