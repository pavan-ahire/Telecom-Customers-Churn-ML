import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide",
    
)
st.image(
    "telecom.png",
    #use_container_width=True,caption=None,width=10
     width=900
)

# ----------------------------------
# Custom CSS Styling
# ----------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(to right, #f8f9fa, #eef2f3);
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background:linear-gradient(90deg, #084298, #0b5ed7);
}

/* Sidebar section headers */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;   /* soft white */
    font-weight: 600;
}
            
/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 500;
}

/* INPUT TEXT COLOR (VERY IMPORTANT FIX) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] textarea {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Placeholder text */
section[data-testid="stSidebar"] input::placeholder {
    color: #6c757d !important;
}

/* Dropdown selected value */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Dropdown options */
div[data-baseweb="popover"] {
    color: #000000 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    padding: 0.6em 1.2em;
    border-radius: 10px;
    border: none;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    background: linear-gradient(to right, #dd2476, #ff512f);
    transform: scale(1.03);
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #dcdcdc;
}

/* ===============================
   FIX SLIDER LABEL WRAPPING
================================*/

/* Ensure slider labels stay horizontal */
section[data-testid="stSidebar"] .stSlider label {
    white-space: nowrap !important;
    display: block !important;
    width: 100% !important;
}

/* Prevent vertical text rendering */
section[data-testid="stSidebar"] .stSlider {
    min-width: 100% !important;
}

/* Ensure slider container doesn't shrink */
section[data-testid="stSidebar"] .stSlider > div {
    width: 100% !important;
}
/* Slider value text (12) */
section[data-testid="stSidebar"] .stSlider p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}
            
/* Banner image size control */
img {
    max-height: 240px;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)



st.title("📞 Telecom Customer Churn Prediction")
st.write("Predict whether a customer is likely to **churn (Yes / No)** using Machine Learning.")

# ----------------------------------
# Load model, scaler, feature columns
# ----------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("churn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("feature_columns.pkl", "rb"))
    return model, scaler, features

model, scaler, feature_columns = load_artifacts()

# ----------------------------------
# Encoding maps (same logic as training)
# ----------------------------------
binary_map = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}

service_map = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}

internet_service_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_method_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}


# ----------------------------------
# Sidebar - User Inputs
# ----------------------------------
st.sidebar.header("🧾 Customer Information")

def user_input_features():
    display_data = {}
    model_data = {}

    # ===============================
    # 👤 Customer Demographics
    # ===============================
    st.sidebar.markdown("### 👤 Customer Demographics")

    gender = st.sidebar.selectbox("Gender", gender_map.keys())
    display_data["gender"] = gender
    model_data["gender"] = gender_map[gender]

    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    display_data["SeniorCitizen"] = senior
    model_data["SeniorCitizen"] = 1 if senior == "Yes" else 0

    partner = st.sidebar.selectbox("Partner", binary_map.keys())
    display_data["Partner"] = partner
    model_data["Partner"] = binary_map[partner]

    dependents = st.sidebar.selectbox("Dependents", binary_map.keys())
    display_data["Dependents"] = dependents
    model_data["Dependents"] = binary_map[dependents]

    # ===============================
    # 📄 Account Information
    # ===============================
    st.sidebar.markdown("### 📄 Account Information")

    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    display_data["tenure"] = tenure
    model_data["tenure"] = tenure

    contract = st.sidebar.selectbox("Contract", contract_map.keys())
    display_data["Contract"] = contract
    model_data["Contract"] = contract_map[contract]

    paperless = st.sidebar.selectbox("Paperless Billing", binary_map.keys())
    display_data["PaperlessBilling"] = paperless
    model_data["PaperlessBilling"] = binary_map[paperless]

    # ===============================
    # 📡 Service Details
    # ===============================
    st.sidebar.markdown("### 📡 Service Details")

    phone = st.sidebar.selectbox("Phone Service", binary_map.keys())
    display_data["PhoneService"] = phone
    model_data["PhoneService"] = binary_map[phone]

    multiple = st.sidebar.selectbox("Multiple Lines", service_map.keys())
    display_data["MultipleLines"] = multiple
    model_data["MultipleLines"] = service_map[multiple]

    internet = st.sidebar.selectbox("Internet Service", internet_service_map.keys())
    display_data["InternetService"] = internet
    model_data["InternetService"] = internet_service_map[internet]

    online_sec = st.sidebar.selectbox("Online Security", service_map.keys())
    display_data["OnlineSecurity"] = online_sec
    model_data["OnlineSecurity"] = service_map[online_sec]

    online_backup = st.sidebar.selectbox("Online Backup", service_map.keys())
    display_data["OnlineBackup"] = online_backup
    model_data["OnlineBackup"] = service_map[online_backup]

    tech = st.sidebar.selectbox("Tech Support", service_map.keys())
    display_data["TechSupport"] = tech
    model_data["TechSupport"] = service_map[tech]

    tv = st.sidebar.selectbox("Streaming TV", service_map.keys())
    display_data["StreamingTV"] = tv
    model_data["StreamingTV"] = service_map[tv]

    movies = st.sidebar.selectbox("Streaming Movies", service_map.keys())
    display_data["StreamingMovies"] = movies
    model_data["StreamingMovies"] = service_map[movies]

    # ===============================
    # 💳 Billing Information
    # ===============================
    st.sidebar.markdown("### 💳 Billing Information")

    payment = st.sidebar.selectbox("Payment Method", payment_method_map.keys())
    display_data["PaymentMethod"] = payment
    model_data["PaymentMethod"] = payment_method_map[payment]

    monthly = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=50.0)
    display_data["MonthlyCharges"] = monthly
    model_data["MonthlyCharges"] = monthly

    total = st.sidebar.number_input("Total Charges", min_value=0.0, value=100.0)
    display_data["TotalCharges"] = total
    model_data["TotalCharges"] = total

    display_df = pd.DataFrame([display_data])
    model_df = pd.DataFrame([model_data], columns=feature_columns)

    return display_df, model_df



#input_df = user_input_features()
display_df, input_df = user_input_features()


# ----------------------------------
# Prediction
# ----------------------------------
st.subheader("🔍 Prediction Result")

if st.button("Predict Churn"):
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is **Likely to Churn**")
    else:
        st.success(f"✅ Customer is **Not Likely to Churn**")

    st.write(f"📊 **Churn Probability:** `{probability:.2f}`")

    # Risk level
    if probability >= 0.7:
        st.warning("🔴 Risk Level: HIGH")
    elif probability >= 0.4:
        st.info("🟠 Risk Level: MEDIUM")
    else:
        st.success("🟢 Risk Level: LOW")
    
    st.subheader("🧾 Customer Input Summary")

    # Display input values as a table
    st.dataframe(display_df, use_container_width=True,height=100)

    
# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("📊 Machine Learning Powered | Streamlit App | Created by Pavan Ahire")
