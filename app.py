import streamlit as st
import pandas as pd
import joblib
from preprocessing import preprocess_data

def load_model():
    model = joblib.load("ideal_model.pkl")
    return model

trained_columns = joblib.load('trained_columns.pkl')

model = load_model()

st.set_page_config(page_title="🚜 Bulldozer Price Prediction", layout="wide")

st.title("🚜 Bulldozer Price Prediction App")
st.write("Fill in the details of a bulldozer OR upload a CSV file to predict its auction price.")

uploaded_file = st.file_uploader("Upload a CSV file with bulldozer data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview 📄")
    st.dataframe(df)

    st.subheader("Preprocessing the data 🔄")
    processed_df = preprocess_data(df)

    for col in trained_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    processed_df = processed_df[trained_columns]

    st.dataframe(processed_df)

    st.subheader("Predictions 🤑")
    preds = model.predict(processed_df)

    result_df = pd.DataFrame({
        "Predicted Sale Price": preds
    })
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode()
    st.download_button(
        "Download Predictions as CSV",
        data=csv,
        file_name="predicted_prices.csv",
        mime="text/csv"
    )

else:
    st.subheader("Manual Entry 📝")

    with st.form("bulldozer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            SalesID = st.number_input("SalesID", value=0)
            MachineID = st.number_input("MachineID", value=0)
            ModelID = st.number_input("ModelID", value=0)
            datasource = st.number_input("Datasource", value=0)
            auctioneerID = st.number_input("AuctioneerID", value=0)
            YearMade = st.number_input("YearMade", value=2000)
            MachineHoursCurrentMeter = st.number_input("MachineHoursCurrentMeter", value=0)
            UsageBand = st.text_input("UsageBand")
            saledate = st.date_input("Saledate")

        with col2:
            fiModelDesc = st.text_input("fiModelDesc")
            fiBaseModel = st.text_input("fiBaseModel")
            fiSecondaryDesc = st.text_input("fiSecondaryDesc")
            fiModelSeries = st.text_input("fiModelSeries")
            fiModelDescriptor = st.text_input("fiModelDescriptor")
            ProductSize = st.text_input("ProductSize")
            fiProductClassDesc = st.text_input("fiProductClassDesc")
            state = st.text_input("State")
            ProductGroup = st.text_input("ProductGroup")

        with col3:
            ProductGroupDesc = st.text_input("ProductGroupDesc")
            Drive_System = st.text_input("Drive_System")
            Enclosure = st.text_input("Enclosure")
            Forks = st.text_input("Forks")
            Pad_Type = st.text_input("Pad_Type")
            Ride_Control = st.text_input("Ride_Control")
            Stick = st.text_input("Stick")
            Transmission = st.text_input("Transmission")
            Turbocharged = st.text_input("Turbocharged")

        submitted = st.form_submit_button("Predict Price 🚀")

    if submitted:
        input_dict = {
            "SalesID": [SalesID],
            "MachineID": [MachineID],
            "ModelID": [ModelID],
            "datasource": [datasource],
            "auctioneerID": [auctioneerID],
            "YearMade": [YearMade],
            "MachineHoursCurrentMeter": [MachineHoursCurrentMeter],
            "UsageBand": [UsageBand],
            "saledate": [pd.to_datetime(saledate)],
            "fiModelDesc": [fiModelDesc],
            "fiBaseModel": [fiBaseModel],
            "fiSecondaryDesc": [fiSecondaryDesc],
            "fiModelSeries": [fiModelSeries],
            "fiModelDescriptor": [fiModelDescriptor],
            "ProductSize": [ProductSize],
            "fiProductClassDesc": [fiProductClassDesc],
            "state": [state],
            "ProductGroup": [ProductGroup],
            "ProductGroupDesc": [ProductGroupDesc],
            "Drive_System": [Drive_System],
            "Enclosure": [Enclosure],
            "Forks": [Forks],
            "Pad_Type": [Pad_Type],
            "Ride_Control": [Ride_Control],
            "Stick": [Stick],
            "Transmission": [Transmission],
            "Turbocharged": [Turbocharged]
        }

        input_df = pd.DataFrame(input_dict)

        st.subheader("Your Input Data 📋")
        st.dataframe(input_df)

        processed_input = preprocess_data(input_df)

        pred_price = model.predict(processed_input)[0]

        st.success(f"🎯 Predicted Bulldozer Sale Price: **${pred_price:,.2f}**")