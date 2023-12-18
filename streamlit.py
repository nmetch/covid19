# app.py
import streamlit as st
import pandas as pd
from fbprophet import Prophet
import plotly.express as px

# Title of the app
st.title("COVID-19 Forecasting App")

# File uploader widget
uploaded_file = st.file_uploader("COVID-19_Case_Surveillance_Public_Use_Data.csv", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the data into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.write("### COVID-19 Data:")
        st.dataframe(df.head())

        # Preprocess data for Prophet
        df = df.rename(columns={"Date": "ds", "Cases": "y"})
        df["ds"] = pd.to_datetime(df["ds"])

        # Create a Prophet model
        model = Prophet()

        # Fit the model
        model.fit(df)

        # Forecast future data
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        # Display the forecast plot
        st.write("### Forecasted Cases:")
        fig_cases = px.line(forecast, x='ds', y='yhat', title='Forecasted Cases')
        st.plotly_chart(fig_cases)

        # Forecast fatality rate
        st.write("### Fatality Rate Forecast:")
        fatality_rate = df["Fatality Rate"].mean()  # You can replace this with your own calculation
        st.write(f"The forecasted fatality rate is approximately: {fatality_rate:.2%}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("COVID-19_Case_Surveillance_Public_Use_Data.csv")
