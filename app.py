import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# -------------------- LOAD MODEL & PREPROCESSORS --------------------

model = tf.keras.models.load_model("model.h5")
model = tf.keras.models.load_model("model.h5", compile=False)


with open("one_hot_encode_geo.pkl", "rb") as f:
    one_hot_encode_geo = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encode_gender = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------- STREAMLIT UI --------------------

st.title("Customer Churn Prediction")

Geography = st.selectbox(
    "Geography",
    one_hot_encode_geo.categories_[0]
)

Gender = st.selectbox(
    "Gender",
    label_encode_gender.classes_
)

Age = st.slider("Age", 18, 92, 30)
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
Balance = st.number_input("Balance", value=37997.0)
EstimatedSalary = st.number_input("Estimated Salary", value=120000.0)
Tenure = st.slider("Tenure (years)", 0, 10, 4)
NumOfProducts = st.slider("Number of Products", 1, 4, 2)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])

# -------------------- PREDICTION --------------------

if st.button("Predict Churn"):

    # Step 1: Build initial DataFrame (WITH Geography)
    input_data = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [label_encode_gender.transform([Gender])[0]],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary]
    })

    # Step 2: One-hot encode Geography (2D input ONLY)
    geo_encoded = one_hot_encode_geo.transform(
        input_data[["Geography"]]
    )

    geo_encoded_df = pd.DataFrame(
        geo_encoded.toarray() if hasattr(geo_encoded, "toarray") else geo_encoded,
        columns=one_hot_encode_geo.get_feature_names_out()
    )

    # Step 3: Drop Geography & concatenate
    input_data = input_data.drop("Geography", axis=1)

    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Step 4: Enforce exact training feature order
    input_data = input_data[scaler.feature_names_in_]

    # Step 5: Scale
    input_scaled = scaler.transform(input_data)

    # Step 6: Predict
    prediction = model.predict(input_scaled)
    prediction_prob = prediction[0][0]

    # Step 7: Output
    if prediction_prob > 0.5:
        st.error("⚠️ The customer is likely to churn")
    else:
        st.success("✅ The customer is not likely to churn")

    st.write(f"Churn probability: **{prediction_prob:.2f}**")
