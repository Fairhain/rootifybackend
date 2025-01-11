import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class EquationModel:
    def __init__(self, soil_erosion, crop_yield, water, soil_ph):
        self.soil_erosion = soil_erosion
        self.crop_yield = crop_yield
        self.water = water
        self.soil_ph = soil_ph

        self.soil_model = None
        self.crop_model = None
        self.water_model = None
        self.ph_model = None

    def train_model(self, X, y, model_name):
        """
        Helper method to train and save a model.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        joblib.dump(model, model_name)
        return model

    def loop_models(self):
        """
        Train and save models for each target variable.
        """
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])  # Match the number of samples

        # Train and save models
        self.soil_model = self.train_model(X, np.array(self.soil_erosion), "erosion_trend_model.pkl")
        self.crop_model = self.train_model(X, np.array(self.crop_yield), "yield_trend_model.pkl")
        self.water_model = self.train_model(X, np.array(self.water), "water_trend_model.pkl")
        self.ph_model = self.train_model(X, np.array(self.soil_ph), "ph_trend_model.pkl")

    def predict(self, input_features):
        """
        Load models (if not already loaded) and make predictions.
        """
        ret = []

        # Soil Erosion Prediction
        if not self.soil_model:
            try:
                self.soil_model = joblib.load("erosion_trend_model.pkl")
            except FileNotFoundError:
                raise Exception("Erosion model not found. Train the model first.")
        ret.append(self.soil_model.predict([input_features])[0])

        # Crop Yield Prediction
        if not self.crop_model:
            try:
                self.crop_model = joblib.load("yield_trend_model.pkl")
            except FileNotFoundError:
                raise Exception("Crop yield model not found. Train the model first.")
        ret.append(self.crop_model.predict([input_features])[0])

        # Water Prediction
        if not self.water_model:
            try:
                self.water_model = joblib.load("water_trend_model.pkl")
            except FileNotFoundError:
                raise Exception("Water model not found. Train the model first.")
        ret.append(self.water_model.predict([input_features])[0])

        # Soil pH Prediction
        if not self.ph_model:
            try:
                self.ph_model = joblib.load("ph_trend_model.pkl")
            except FileNotFoundError:
                raise Exception("Soil pH model not found. Train the model first.")
        ret.append(self.ph_model.predict([input_features])[0])

        return ret
