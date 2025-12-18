# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("farmer_loan_dataset.csv")

# Encode categorical columns
cat_cols = ["education", "region", "crop_type", "irrigation",
            "past_loan_status", "loan_purpose", "risk_label"]

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Split features and target
X = df.drop("eligible", axis=1)
y = df["eligible"]

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Streamlit UI
st.title("🌾 Farmer Loan Eligibility Support System")
st.markdown("Use this tool to check if a farmer is eligible for a loan based on various parameters.")

# Input form
with st.form(key="loan_form"):
    age = st.number_input("Age", 18, 80, 30)
    education = st.selectbox("Education Level", le_dict["education"].classes_)
    land_size = st.number_input("Land Size (acres)", 0.1, 50.0, 1.0)
    region = st.selectbox("Region", le_dict["region"].classes_)
    crop_type = st.selectbox("Crop Type", le_dict["crop_type"].classes_)
    irrigation = st.selectbox("Irrigation Type", le_dict["irrigation"].classes_)
    soil_index = st.slider("Soil Health Index", 0, 100, 50)
    avg_yield = st.number_input("Average Yield (tons per acre)", 0.1, 10.0, 1.0)
    income = st.number_input("Annual Income (INR)", 5000, 1000000, 50000)
    past_loan = st.selectbox("Past Loan Status", le_dict["past_loan_status"].classes_)
    credit_score = st.slider("Credit Score", 300, 850, 500)
    loan_requested = st.number_input("Loan Requested (INR)", 5000, 1000000, 50000)
    loan_purpose = st.selectbox("Loan Purpose", le_dict["loan_purpose"].classes_)
    risk_label = st.selectbox("Risk Label", le_dict["risk_label"].classes_)
    
    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    # Prepare input for prediction
    input_dict = {
        "age": age,
        "education": le_dict["education"].transform([education])[0],
        "land_size_acres": land_size,
        "region": le_dict["region"].transform([region])[0],
        "crop_type": le_dict["crop_type"].transform([crop_type])[0],
        "irrigation": le_dict["irrigation"].transform([irrigation])[0],
        "soil_health_index": soil_index,
        "avg_yield_t_per_acre": avg_yield,
        "annual_income_inr": income,
        "income_per_acre": income / land_size,
        "past_loan_status": le_dict["past_loan_status"].transform([past_loan])[0],
        "credit_score": credit_score,
        "loan_requested_inr": loan_requested,
        "loan_purpose": le_dict["loan_purpose"].transform([loan_purpose])[0],
        "requested_to_income_ratio": loan_requested / (income + 1),
        "risk_score_continuous": 0,  # Optional
        "risk_label": le_dict["risk_label"].transform([risk_label])[0]
    }

    input_df = pd.DataFrame([input_dict])

    # Predict eligibility
    prediction = rf_model.predict(input_df)[0]
    st.subheader("✅ Loan Eligibility Prediction:")
    st.success("Eligible ✅" if prediction == 1 else "Not Eligible ❌")
