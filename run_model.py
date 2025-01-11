from water_model import WaterModel

# Create an instance of the model
model_instance = WaterModel("TARP.csv")

model_instance.train_model()

# Example input: Replace with actual feature values
new_input = [54, 22, 70, 21, 19.52, 2.13, 55.04, 6.3, 101.5, 6.5, 202.93, 90, 42, 43]

# Predict water required
predicted_water = model_instance.predict(new_input)
print("Predicted Water Required:", str(predicted_water) + " oz")
