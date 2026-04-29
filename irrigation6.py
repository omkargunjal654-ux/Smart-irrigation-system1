import streamlit as st
import pandas as pd
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
            df[column] = le.fit_transform(df[column])
            encoders[column] = le

    st.subheader("Encoded Dataset")
    st.write(df.head())

    # Target column
    target_column = st.selectbox("Select Target Column", df.columns)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

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
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    st.subheader("Model Accuracy")
    st.write(f"Decision Tree Accuracy: {dt_acc:.2f}")
    st.write(f"Random Forest Accuracy: {rf_acc:.2f}")

    # Prediction section
    st.subheader("Predict Irrigation")
    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

    input_df = pd.DataFrame([user_input])

    model_choice = st.selectbox(
        "Choose Model", ["Decision Tree", "Random Forest"]
    )

    if st.button("Predict"):
        if model_choice == "Decision Tree":
            prediction = dt_model.predict(input_df)[0]
        else:
            prediction = rf_model.predict(input_df)[0]
        st.success(f"Prediction Result: {prediction}")

else:
    st.info("Please upload a CSV dataset to continue.")
