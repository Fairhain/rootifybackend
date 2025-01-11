from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib  # Assuming your model is saved using joblib or pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from your React frontend

# Load the pre-trained model
model = joblib.load("water_prediction_model.pkl")  # Update with your model file path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the weather data from the request
        data = request.json
        features = np.array(data['features']).reshape(1, -1)  # Assuming a single row of input

        # Make a prediction
        prediction = model.predict(features)
        
        # Return the prediction
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
