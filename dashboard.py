import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Telecom Customer Churn Dashboard",
    page_icon="📞",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.autolayout"] = True

# ================= LOAD DATA =================
df = pd.read_csv("telecom_dashboard.csv")

# ================= DATA PREP =================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_Num"] = np.where(df["Churn"] == "Yes", 1, 0)
df["Churn_Label"] = np.where(df["Churn"] == "Yes", "Churned", "Retained")

num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "InternetService", "Contract", "PaymentMethod", "Churn_Label"
]

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.kpi-box {
    border: 1.5px solid #e0e0e0;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    background-color: #ffffff;
}
.kpi-title {
    font-size: 14px;
    color: #6c757d;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>📞 Telecom Customer Churn EDA Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= KPI SECTION =================
total_customers = df.shape[0]
churned = df[df["Churn"] == "Yes"].shape[0]
churn_rate = round((churned / total_customers) * 100)

avg_monthly = round(df["MonthlyCharges"].mean(), 2)
avg_tenure = round(df["tenure"].mean(), 1)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Total Customers</div>
    <div class="kpi-value">{total_customers}</div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Churned Customers</div>
    <div class="kpi-value">{churned}</div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Churn Rate (%)</div>
    <div class="kpi-value">{churn_rate}%</div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg Monthly Charges</div>
    <div class="kpi-value">₹ {avg_monthly}</div>
</div>
""", unsafe_allow_html=True)

k5.markdown(f"""
<div class="kpi-box">
    <div class="kpi-title">Avg Tenure (Months)</div>
    <div class="kpi-value">{avg_tenure}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🛠️ Analysis Controls")

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

# ================= UNIVARIATE ANALYSIS =================
if analysis_type == "Univariate Analysis":

    st.subheader("📊 Univariate Analysis")

    feature = st.sidebar.selectbox(
        "Select Feature",
        num_features + ["Churn_Label"]
    )

    col1, col2 = st.columns(2, gap="large")

    if feature in num_features:
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[feature], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[feature], ax=ax)
            ax.set_title(f"Boxplot of {feature}")
            st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[feature], ax=ax)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)

# ================= BIVARIATE ANALYSIS =================
else:

    st.subheader("📈 Bivariate Analysis")

    bi_type = st.sidebar.selectbox(
        "Select Relationship",
        ["Num vs Num", "Num vs Cat", "Cat vs Cat"]
    )

    if bi_type == "Num vs Num":
        x = st.sidebar.selectbox("X Axis", num_features)
        y = st.sidebar.selectbox("Y Axis", num_features, index=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="Churn_Label",
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    elif bi_type == "Num vs Cat":
        num = st.sidebar.selectbox("Numerical Feature", num_features)
        cat = st.sidebar.selectbox("Categorical Feature", cat_features)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=df,
            x=cat,
            y=num,
            errorbar=None,
            ax=ax
        )
        ax.set_title(f"Average {num} by {cat}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    else:
        x = st.sidebar.selectbox("Category", cat_features)
        y = st.sidebar.selectbox("Hue", cat_features, index=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(data=df, x=x, hue=y, ax=ax)
        ax.set_title(f"{x} vs {y}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

# ================= CORRELATION =================
st.markdown("---")
st.subheader("🏆 Feature Correlation with Churn")

corr_df = (
    df[num_features + ["Churn_Num"]]
    .corr()["Churn_Num"]
    .drop("Churn_Num")
    .reset_index()
)

corr_df.columns = ["Feature", "Correlation"]
corr_df["Abs_Correlation"] = corr_df["Correlation"].abs()
corr_df = corr_df.sort_values("Abs_Correlation", ascending=False)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.dataframe(corr_df[["Feature", "Correlation"]], use_container_width=True)

with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=corr_df,
        x="Abs_Correlation",
        y="Feature",
        ax=ax
    )
    ax.set_title("Absolute Correlation with Churn")
    st.pyplot(fig)

st.warning(
    "Churn is influenced by multiple factors together. "
    "No single feature alone determines customer churn."
)
