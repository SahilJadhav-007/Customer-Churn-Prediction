
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained ML pipeline
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.joblib")

model = load_model()

# Application title
st.title("Customer Churn Prediction")

# Description
st.write(
    "This application predicts whether a customer "
    "is likely to leave the service using Machine Learning."
)



# --------------------------------
# 1. Customer Demographics
# --------------------------------

st.subheader("Customer Demographics")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


# --------------------------------
# 2. Customer Services
# --------------------------------

st.subheader("Customer Services")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------
# 3. Account Information
# --------------------------------

st.subheader("Account Information")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0,
        step=1.0
    )



# --------------------------------
# 4. Prepare Customer Data
# --------------------------------

input_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Convert dictionary into a DataFrame
customer_df = pd.DataFrame([input_data])



# --------------------------------
# 5. Predict Customer Churn
# --------------------------------

st.subheader("Churn Prediction")

if st.button("Predict Customer Churn"):

    # Predict customer churn
    prediction = model.predict(customer_df)

    # Get churn probabilities
    probabilities = model.predict_proba(customer_df)

    # Find the probability corresponding to "Yes"
    classes = model.named_steps["classifier"].classes_
    yes_index = list(classes).index("Yes")

    churn_probability = probabilities[0][yes_index]

    # Display prediction
    if prediction[0] == "Yes":
        st.error("This customer is likely to churn!")
    else:
        st.success("This customer is unlikely to churn!")

    # Display probability
    st.write(
        f"Churn Probability: {churn_probability:.2%}"
    )

    st.progress(float(churn_probability))
