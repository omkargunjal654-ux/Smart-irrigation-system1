import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("Smart Irrigation System using Machine Learning")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Encode categorical columns
    encoders = {}
    for column in df.columns:
        if df[column].dtype == "object":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = le

    # Convert all columns to numeric, coercing errors to NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN after conversion
    df = df.dropna(axis=1, how="all")

    # Fill any remaining NaN values with column mean
    df = df.fillna(df.mean())

    st.subheader("Encoded Dataset")
    st.write(df.head())

    # Target column
    target_column = st.selectbox("Select Target Column", df.columns)
    X = df.drop(target_column, axis=1)
    y = df[target_column].astype(int)

    # Ensure X is float64
    X = X.astype(np.float64)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    st.subheader("Model Accuracy")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree Accuracy", f"{dt_acc * 100:.2f}%")
    col2.metric("Random Forest Accuracy", f"{rf_acc * 100:.2f}%")

    # Prediction section
    st.subheader("Predict Irrigation")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            f"Enter {col}",
            value=float(X[col].mean()),
            format="%.4f"
        )

    input_df = pd.DataFrame([user_input]).astype(np.float64)

    model_choice = st.selectbox(
        "Choose Model", ["Decision Tree", "Random Forest"]
    )

    if st.button("Predict"):
        if model_choice == "Decision Tree":
            prediction = dt_model.predict(input_df)[0]
        else:
            prediction = rf_model.predict(input_df)[0]

        # Decode prediction if target was originally categorical
        if target_column in encoders:
            prediction_label = encoders[target_column].inverse_transform([int(prediction)])[0]
        else:
            prediction_label = prediction

        st.success(f"Prediction Result: {prediction_label}")

else:
    st.info("Please upload a CSV dataset to continue.")
