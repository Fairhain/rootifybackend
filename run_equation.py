from equation_model import EquationModel

model = EquationModel(([6, 7, 8, 9, 10]), ([6, 7, 8, 9, 10]),([6, 7, 8, 9, 10]),([6, 7, 8, 9, 10]))

model.loop_models()

erosion = []
crop = []
water = []
ph = []


for i in range(1, 6):
    predicted_values = model.predict([i])
    print(predicted_values)
    erosion.append(float(predicted_values[0]))
    
    crop.append(float(predicted_values[1]))

    water.append(float(predicted_values[2]))

    ph.append(float(predicted_values[3]))

print("*****************")
print(erosion)
print(crop)
print(water)
print(ph)





