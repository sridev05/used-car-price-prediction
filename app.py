import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# Load Custom CSS (with cache buster)
# ===============================
import time

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        with open(css_path) as f:
            css_content = f.read()
            # Add timestamp comment to force browser reload
            cache_buster = f"/* Cache Buster: {time.time()} */"
            st.markdown(f'<style>{cache_buster}\n{css_content}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("style.css file not found! Please ensure it's in the same directory as app.py")

load_css()

# ===============================
# Set up relative paths
# ===============================
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "models", "used_car_price_model.pkl")
encoders_path = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# ===============================
# Load model and encoders
# ===============================
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    return model, label_encoders

model, label_encoders = load_model_and_encoders()

# ===============================
# HTML Header
# ===============================
st.markdown("""
    <div class="header-container">
        <div class="icon">🚗</div>
        <h1 class="main-title">Used Car Price Predictor</h1>
        <p class="subtitle">Get instant price estimates using AI-powered predictions</p>
    </div>
""", unsafe_allow_html=True)

# ===============================
# Main Content - Two Columns
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📋 Car Details</h3>', unsafe_allow_html=True)
    
    brand = st.selectbox("Brand", label_encoders['Brand'].classes_, key="brand")
    model_name = st.selectbox("Model", label_encoders['model'].classes_, key="model")
    fuel_type = st.selectbox("Fuel Type", label_encoders['FuelType'].classes_, key="fuel")
    transmission = st.selectbox("Transmission", label_encoders['Transmission'].classes_, key="trans")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🔧 Specifications</h3>', unsafe_allow_html=True)
    
    year = st.slider("Year of Manufacture", min_value=1980, max_value=2025, value=2015, key="year")
    age = st.slider("Car Age (years)", min_value=0, max_value=50, value=5, key="age")
    km_driven = st.text_input("Kilometers Driven", value="85000", key="km")
    owner = st.selectbox("Owner Type", label_encoders['Owner'].classes_, key="owner")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Prediction Button and Result
# ===============================
st.markdown('<div class="button-container">', unsafe_allow_html=True)
predict_button = st.button("🔮 Predict Price", key="predict", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if predict_button:
    try:
        # Clean kmDriven input
        km_driven_clean = float(str(km_driven).replace(',', '').replace(' km', '').replace(' ', ''))
        
        # Create DataFrame
        input_df = pd.DataFrame({
            'Brand': [brand],
            'model': [model_name],
            'Year': [year],
            'Age': [age],
            'kmDriven': [km_driven_clean],
            'Transmission': [transmission],
            'Owner': [owner],
            'FuelType': [fuel_type]
        })
        
        # Encode categorical features
        for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
        
        # Predict price
        price = model.predict(input_df)[0]
        
        # Add smooth scroll to result
        st.markdown("""
            <script>
                setTimeout(function() {
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                }, 200);
            </script>
        """, unsafe_allow_html=True)
        
        # Display result with animation
        st.markdown(f"""
            <div class="result-container">
                <div class="result-card">
                    <div class="result-icon">💰</div>
                    <h2 class="result-title">Predicted Selling Price</h2>
                    <div class="result-price">₹ {int(price):,}</div>
                    <p class="result-note">*This is an estimated price based on market trends</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
            <div class="error-container">
                <div class="error-card">
                    <div class="error-icon">⚠️</div>
                    <h3>Prediction Error</h3>
                    <p>{str(e)}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("""
    <div class="footer">
        <p>Powered by Machine Learning | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)