import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

class WaterModel:
    def __init__(self, data_path:str):
        self.data_path = data_path
        self.model = None

    def data(self):
        # Load the dataset
        df = pd.read_csv(self.data_path)

        # Encode the 'Status' column (ON = 1, OFF = 0)
        df['Status'] = df['Status'].apply(lambda x: 1 if x == 'ON' else 0)

        # Define a target soil moisture level
        df['Target Moisture'] = 65

        # Calculate the water required to reach the target soil moisture
        df['WaterRequired'] = (
            0.7 * (df['Target Moisture'] - df['Soil Moisture']) +  # Base contribution from Soil Moisture
            0.3 * df['Temperature'] +  # Add Temperature with weight 0.3
            0.25 * df[' Soil Humidity'] +  # Add Soil Humidity with weight 0.25
            0.15 * df['Time'] +  # Add Time with weight 0.15
            0.3 * df['Air temperature (C)'] -  # Add Air temperature with weight 0.2
            0.1 * df['Air humidity (%)'] +  # Subtract Air humidity with weight 0.1
            0.04 * df['Wind speed (Km/h)'] +  # Add Wind speed with weight 0.05
            0.04 * df['Wind gust (Km/h)'] +  # Add Wind gust with weight 0.04
            0.03 * df['Pressure (KPa)'] +  # Add Pressure with weight 0.03
            0.02 * df['ph'] +  # Add ph with weight 0.02
            0.01 * df['rainfall']  # Add Rainfall with weight 0.01
        )

        # Add random noise to WaterRequired
        np.random.seed(42)
        df['WaterRequired'] += np.random.normal(0, 5, size=len(df))

        # Check for NaN values and handle them
        df['WaterRequired'].fillna(df['WaterRequired'].mean(), inplace=True)

        # Drop unnecessary columns
        df = df.drop(['Status', 'Target Moisture'], axis=1)

        # Ensure no NaN values in the entire dataset
        df.fillna(0, inplace=True)

        # Separate features (X) and target (y)
        x = df.drop(['WaterRequired'], axis=1)  # Features: all columns except WaterRequired
        y = df['WaterRequired']  

        return x, y

    def train_model(self):
        X, y = self.data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the Random Forest Regressor
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        print("R-squared Score:", r2_score(y_test, y_pred))     

        joblib.dump(self.model, "water_prediction_model.pkl")


    def predict(self, input_features):
        # Ensure the model is trained or loaded
        try:
            self.model = joblib.load("water_prediction_model.pkl")
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")
            
        # Make prediction
        prediction = self.model.predict([input_features])
        return prediction[0]
