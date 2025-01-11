from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib 
import numpy as np
from equation_model import EquationModel



app = Flask(__name__)
CORS(app) 

# Load the pre-trained model
model = joblib.load("water_prediction_model.pkl")  
erosion_model = joblib.load("erosion_trend_model.pkl")
yield_model = joblib.load("yield_trend_model.pkl")
water_model = joblib.load("water_trend_model.pkl")
ph_model = joblib.load("ph_trend_model.pkl")


@app.route('/predict', methods=['POST'])
def predict():
    try:        
        data = request.json
        features = np.array(data['features']).reshape(1, -1) 

        prediction = model.predict(features)
        
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/graph', methods=['POST'])
def graph():
    try:
        data = request.json
        print("Received data:", data) 
        
        required_keys = ['erosion', 'crop', 'water', 'ph']
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing required keys in input data"}), 400

        model = EquationModel(data['erosion'], data['crop'], data['water'], data['ph'])
        model.loop_models()

        erosion = []
        crop = []
        water = []
        ph = []

        for i in range(1, 7):
            predicted_values = model.predict([i])
            erosion.append(float(predicted_values[0]))
            crop.append(float(predicted_values[1]))
            water.append(float(predicted_values[2]))
            ph.append(float(predicted_values[3]))

        return jsonify({
            "erosion": erosion,
            "crop": crop,
            "water": water,
            "ph": ph
        })

    except Exception as e:
        print("Error:", e)  
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
