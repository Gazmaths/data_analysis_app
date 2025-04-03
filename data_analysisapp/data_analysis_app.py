import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Data Explorer", layout="wide")

# ========== STYLING ==========
st.markdown("""
    <style>
        .main {background-color: #fafafa;}
        h1 {color: #3E64FF;}
        .stButton>button {
            color: white;
            background-color: #3E64FF;
            padding: 0.5rem 1rem;
            border-radius: 10px;
        }
        .stSidebar [data-testid="stImage"] {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.title("📊 Interactive Data Analysis App")
st.markdown("Use this app to **explore, visualize**, and **model** your data with ease.")

# ========== FILE UPLOADER ==========
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=150)
st.sidebar.header("📁 Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# ========== APP LOGIC ==========
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    st.success("✅ File uploaded successfully!")

    with st.expander("👁️ Data Preview"):
        st.dataframe(df.head())

    with st.expander("📈 Dataset Summary"):
        st.dataframe(df.describe())

    # Sidebar column selection
    st.sidebar.header("🛠️ Column Selection")
    selected_columns = st.sidebar.multiselect("Select columns to include in analysis:", df.columns.tolist(), default=df.columns.tolist())

    if selected_columns:
        # ========== Correlation Heatmap ==========
        st.subheader("🔗 Correlation Matrix")
        correlation = df[selected_columns].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        st.markdown("---")

        # ========== Interactive Scatter Plot ==========
        st.subheader("📌 Interactive Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", selected_columns)
        with col2:
            y_axis = st.selectbox("Y-axis", selected_columns)
        fig_plot = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig_plot, use_container_width=True)

        st.markdown("---")

        # ========== Histogram ==========
        st.subheader("📊 Histogram")
        column_to_plot = st.selectbox("Select a column for histogram", selected_columns)
        fig_hist, ax_hist = plt.subplots()
        df[column_to_plot].hist(ax=ax_hist, bins=20, color='#3E64FF')
        ax_hist.set_title(f"Histogram of {column_to_plot}")
        st.pyplot(fig_hist)

        st.markdown("---")

        # ========== Linear Regression ==========
        st.subheader("📉 Linear Regression Model")
        st.markdown("Select a **target variable** and one or more **feature columns** below:")
        target = st.selectbox("🎯 Target Variable", options=selected_columns)
        features = st.multiselect("📥 Feature(s)", [col for col in selected_columns if col != target], default=[col for col in selected_columns if col != target])

        if features:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)

            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.success(f"📐 Mean Squared Error: `{mse:.4f}`")

            coef_df = pd.DataFrame({"Feature": features, "Coefficient": reg.coef_})
            st.write("📋 **Regression Coefficients**")
            st.dataframe(coef_df)

            # Regression plot
            st.write("📉 **Regression Plot** (Actual vs Predicted)")
            fig_reg, ax_reg = plt.subplots()
            sns.regplot(x=y_test, y=y_pred, ax=ax_reg, line_kws={"color": "red"})
            ax_reg.set_xlabel("Actual")
            ax_reg.set_ylabel("Predicted")
            ax_reg.set_title("Actual vs Predicted")
            st.pyplot(fig_reg)

else:
    st.info("⬅️ Upload a CSV file to begin.")

