import joblib

try:
    erosion_model = joblib.load("erosion_trend_model.pkl")
    print("Model loaded successfully")
except Exception as e:
    print("Error:", e)
