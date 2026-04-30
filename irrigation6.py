import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Smart Irrigation System using Machine Learning")

# ── Load dataset directly (no upload needed) ──────────────────────────────
DATASET_PATH = "irrigation_dataset(5).csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(DATASET_PATH)
except FileNotFoundError:
    st.error(
        f"Dataset file '{DATASET_PATH}' not found. "
        "Make sure it is in the same folder as this script."
    )
    st.stop()

st.subheader("Dataset Preview")
st.write(df.head())

# ── Step 1: Manually encode Irrigation target with fixed mapping ──────────
TARGET = "Irrigation"
if TARGET not in df.columns:
    st.error("Column 'Irrigation' not found in dataset.")
    st.stop()

df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
df[TARGET] = df[TARGET].map({"OFF": 0, "ON": 1})

# ── Step 2: Encode remaining categorical feature columns ──────────────────
encoders = {}
for column in df.columns:
    if column == TARGET:
        continue
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        encoders[column] = le

# ── Step 3: Convert everything to numeric ─────────────────────────────────
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="all")
df = df.fillna(df.mean())

st.subheader("Encoded Dataset")
st.write(df.head())

# ── Step 4: Split features and target ────────────────────────────────────
X = df.drop(TARGET, axis=1).astype(np.float64)
y = df[TARGET].astype(int)

st.info(f"Target distribution — ON: {(y == 1).sum()}  |  OFF: {(y == 0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Step 5: Train models ──────────────────────────────────────────────────
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

st.subheader("Model Accuracy")
col1, col2 = st.columns(2)
col1.metric("Decision Tree Accuracy", f"{dt_acc * 100:.2f}%")
col2.metric("Random Forest Accuracy", f"{rf_acc * 100:.2f}%")

# ── Step 6: Prediction inputs ─────────────────────────────────────────────
st.subheader("Predict Irrigation")
st.markdown("Enter field values below to predict whether irrigation should be **ON** or **OFF**.")

user_input = {}
grid = st.columns(2)
for i, col in enumerate(X.columns):
    with grid[i % 2]:
        user_input[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean()),
            format="%.4f"
        )

input_df = pd.DataFrame([user_input]).astype(np.float64)

model_choice = st.selectbox("Choose Model", ["Decision Tree", "Random Forest"])

if st.button("Predict Irrigation", use_container_width=True):
    model = dt_model if model_choice == "Decision Tree" else rf_model
    prediction = model.predict(input_df)[0]
    label = "ON" if prediction == 1 else "OFF"

    if label == "ON":
        st.success("💧 Irrigation Required: **ON**")
        st.markdown(
            "<div style='background-color:#d4edda;padding:24px;border-radius:12px;"
            "text-align:center;font-size:36px;font-weight:bold;color:#155724;'>"
            "💧 IRRIGATION: ON</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("🚫 Irrigation Not Required: **OFF**")
        st.markdown(
            "<div style='background-color:#fff3cd;padding:24px;border-radius:12px;"
            "text-align:center;font-size:36px;font-weight:bold;color:#856404;'>"
            "🚫 IRRIGATION: OFF</div>",
            unsafe_allow_html=True,
        )
