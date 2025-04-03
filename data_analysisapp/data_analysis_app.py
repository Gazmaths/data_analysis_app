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
st.set_page_config(page_title="Interactive Data Analysis App", layout="wide")

# ========== STYLING ==========
st.markdown("""
    <style>
        h1, h2, h3 {
            color: #3E64FF;
        }
        .stButton>button {
            color: white;
            background-color: #3E64FF;
            padding: 0.4rem 1rem;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.title("📊 Interactive Data Analysis App")
st.markdown("Upload your CSV, visualize trends, explore relationships, and run a linear regression model—all in one place!")

# ========== FILE UPLOADER ==========
st.sidebar.header("📁 Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# ========== DEDUPLICATE COLUMNS ==========
def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols

# ========== MAIN LOGIC ==========
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = deduplicate_columns(df.columns)  # Ensure unique column names
    st.success("✅ File uploaded successfully!")

    st.header("👁️ Data Overview")
    st.dataframe(df.head())

    st.header("📊 Dataset Summary")
    st.dataframe(df.describe())

    # Sidebar column selection
    st.sidebar.header("🛠️ Select Fields for Analysis")
    selected_columns = st.sidebar.multiselect("Choose columns to explore:", df.columns.tolist(), default=df.columns.tolist())

    if selected_columns:
        # ===== Correlation Matrix =====
        st.markdown("---")
        st.subheader("🔗 Correlation Matrix")
        correlation = df[selected_columns].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # ===== Plotly Interactive Scatter Plot =====
        st.markdown("---")
        st.subheader("📌 Interactive Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", options=selected_columns, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y-axis", options=selected_columns, key="scatter_y")
        fig_plot = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"{x_axis} vs {y_axis}")
        st.plotly_chart(fig_plot, use_container_width=True)

        # ===== Histogram =====
        st.markdown("---")
        st.subheader("📊 Histogram")
        column_to_plot = st.selectbox("Select column for histogram:", selected_columns, key="hist_col")
        fig_hist, ax_hist = plt.subplots()
        df[column_to_plot].hist(ax=ax_hist, bins=20, color='#3E64FF')
        ax_hist.set_title(f"Histogram of {column_to_plot}")
        st.pyplot(fig_hist)

        # ===== Linear Regression =====
        st.markdown("---")
        st.subheader("📉 Linear Regression Model")
        st.markdown("Select a **target** and one or more **feature(s)** to build a model.")

        target = st.selectbox("🎯 Select Target Variable", options=selected_columns, key="target_var")
        feature_options = [col for col in selected_columns if col != target]
        features = st.multiselect("📥 Select Feature(s)", options=feature_options, default=feature_options, key="features")

        if features:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            reg = LinearRegression()
            reg.fit(X_train, y_train)

            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.success(f"📐 Mean Squared Error (MSE): `{mse:.4f}`")

            coef_df = pd.DataFrame({
                "Feature": features,
                "Coefficient": reg.coef_
            })
            st.write("📋 **Regression Coefficients**")
            st.dataframe(coef_df)

            # Regression actual vs predicted plot
            st.subheader("📈 Regression Plot: Actual vs Predicted")
            fig_reg, ax_reg = plt.subplots()
            sns.regplot(x=y_test, y=y_pred, ax=ax_reg, line_kws={"color": "red"})
            ax_reg.set_xlabel("Actual")
            ax_reg.set_ylabel("Predicted")
            ax_reg.set_title("Actual vs Predicted Values")
            st.pyplot(fig_reg)

else:
    st.info("📂 Please upload a CSV file to begin.")


