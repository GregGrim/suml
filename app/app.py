# SUML Delivery project series
# By Hryhorii Hrymailo s27157

import streamlit as st
from predict import predict

def app():
    st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸")

    st.title("🌸 Iris Species Predictor")
    st.write("Enter the flower measurements below to predict the Iris species.")

    # Numeric inputs
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, step=0.1)

    if st.button("Predict"):
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = predict(features)
        st.success(f"🌼 Predicted Iris species: **{prediction}**")


if __name__ == "__main__":
    app()