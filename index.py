import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
loaded_model = joblib.load("Model/random_forest_model.pkl")
loaded_scaler = joblib.load("Model/scaler.pkl")

# Streamlit App
st.title("Wine Quality Prediction App")

# Input form untuk fitur-fitur anggur
fixed_acidity = st.slider("Fixed Acidity", min_value=4.0, max_value=16.0, value=8.0)
volatile_acidity = st.slider("Volatile Acidity", min_value=0.1, max_value=2.0, value=0.5)
citric_acid = st.slider("Citric Acid", min_value=0.0, max_value=1.0, value=0.3)
residual_sugar = st.slider("Residual Sugar", min_value=0.5, max_value=15.0, value=2.0)
chlorides = st.slider("Chlorides", min_value=0.01, max_value=0.2, value=0.08)
free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", min_value=1, max_value=72, value=30)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", min_value=6, max_value=289, value=150)
density = st.slider("Density", min_value=0.987, max_value=1.010, value=0.996)
pH = st.slider("pH", min_value=2.7, max_value=4.0, value=3.0)
sulphates = st.slider("Sulphates", min_value=0.3, max_value=2.0, value=0.8)
alcohol = st.slider("Alcohol", min_value=8.0, max_value=15.0, value=10.0)

# Menggunakan model untuk membuat prediksi
input_data = {
    "fixed_acidity": fixed_acidity,
    "volatile_acidity": volatile_acidity,
    "citric_acid": citric_acid,
    "residual_sugar": residual_sugar,
    "chlorides": chlorides,
    "free_sulfur_dioxide": free_sulfur_dioxide,
    "total_sulfur_dioxide": total_sulfur_dioxide,
    "density": density,
    "pH": pH,
    "sulphates": sulphates,
    "alcohol": alcohol
}

# Preprocessing input data menggunakan scaler
scaled_input_data = loaded_scaler.transform(pd.DataFrame([input_data]))

# Membuat prediksi menggunakan model
prediction = loaded_model.predict(scaled_input_data)

# Menampilkan hasil prediksi
st.subheader("Hasil Prediksi:")
if prediction[0] == 1:
    st.write("Anggur berkualitas bagus!")
else:
    st.write("Anggur berkualitas buruk.")
