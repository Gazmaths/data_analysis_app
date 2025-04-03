import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# App title
st.title("Interactive Data Analysis App")

# Upload data section
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Show basic data info
    st.write("### Data Overview")
    st.write(df.head())

    # Display basic statistics of the dataset
    st.write("### Dataset Summary")
    st.write(df.describe())
    
    # Select columns for analysis
    st.sidebar.header("Select Fields for Analysis")
    selected_columns = st.sidebar.multiselect("Select Columns", df.columns.tolist(), default=df.columns.tolist())

    # Correlation heatmap
    st.write("### Correlation Matrix")
    correlation = df[selected_columns].corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(correlation, annot=True, ax=ax_corr, cmap="coolwarm")
    st.pyplot(fig_corr)

    # Interactive Plot (Plotly)
    st.write("### Interactive Plot")
    x_axis = st.selectbox("Select X-axis for plotting", options=selected_columns)
    y_axis = st.selectbox("Select Y-axis for plotting", options=selected_columns)
    fig_plot = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig_plot)

    # Histogram
    st.write("### Histogram")
    column_to_plot = st.selectbox("Select a column to plot histogram", options=selected_columns)
    fig_hist, ax_hist = plt.subplots()
    df[column_to_plot].hist(ax=ax_hist, bins=20)
    ax_hist.set_title(f"Histogram of {column_to_plot}")
    st.pyplot(fig_hist)

    # Linear Regression Section using sns.regplot
    st.write("### Linear Regression Model")
    st.write("Select target variable and feature(s) to perform linear regression.")
    target = st.selectbox("Select Target Variable", options=selected_columns)
    features = st.multiselect("Select Feature(s)", selected_columns, default=[x for x in selected_columns if x != target])

    if len(features) > 0:
        X = df[features]
        y = df[target]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")

        # Display the regression coefficients
        coef_df = pd.DataFrame({"Features": features, "Coefficients": reg.coef_})
        st.write("### Regression Coefficients")
        st.write(coef_df)

        # Show regression plot using sns.regplot
        st.write("### Regression Plot (Seaborn's regplot)")
        fig_reg, ax_reg = plt.subplots()
        sns.regplot(x=y_test, y=y_pred, ax=ax_reg)
        ax_reg.set_xlabel("Actual")
        ax_reg.set_ylabel("Predicted")
        ax_reg.set_title("Actual vs Predicted")
        st.pyplot(fig_reg)

else:
    st.write("Please upload a CSV file to proceed.")

