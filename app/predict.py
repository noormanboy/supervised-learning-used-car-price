import joblib
import pandas as pd

model = joblib.load("../model/car_price_model.pkl")

data = {
    "year": [2019],
    "km_driven": [30000],
    "engine": [1500],
    "fuel_type_Petrol": [1],
    "fuel_type_Diesel": [0],
    "transmission_Manual": [0],
    "transmission_Automatic": [1],
    "owner_type_First": [1],
    "owner_type_Second": [0],
    "owner_type_Third": [0],
}

df = pd.DataFrame(data)

prediction = model.predict(df)

print("Predicted Price:", prediction[0])
