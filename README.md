# Used Car Price Prediction 🚗

A machine learning–based web application that predicts the selling price of a used car using features such as brand, model year, fuel type, transmission, mileage, and ownership details. The application is built using Streamlit for an interactive web interface.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit

## Project Structure
- `app.py` – Complete application (model training + prediction)
- `models/` – Stores trained model files (not committed to Git)
- `data/` – Dataset used for training

## How to Run the Application
pip install -r requirements.txt
streamlit run app.py

## Run the Model
The machine learning model is trained automatically when the application runs.
Model training and prediction logic are handled inside app.py, and the trained model is saved locally in the models/ directory.
