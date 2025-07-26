from flask import Flask, jsonify, request

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a mi API de predicciones de adicción a las redes sociales"

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    Age = request.args.get('Age', None)
    Continent = request.args.get('Continent', None)
    Sleep = request.args.get('Sleep_Hours_Per_Night', None)

    
    columnas = [  "Continent_Asia",  "Continent_Europe",  "Continent_North America",
    "Continent_Oceania",  "Continent_South America"]
    
    continent_dummies = {col: 0 for col in columnas}
    selected_col = f"Continent_{Continent}"
    
    if selected_col in continent_dummies: 
        continent_dummies[selected_col]=1
    else :
        return f"Error: Continente '{Continent}' no reconocido"

    if Age is None or Continent is None or Sleep is None:
        return "Faltan argumentos, no se puede hacer predicciones"
    else:
        input_vector = [Age] + list(continent_dummies.values()) + [Sleep]
        prediction = model.predict([input_vector])
    
    return jsonify({'predictions': prediction[0]})


'''
# Enruta la funcion al endpoint /api/v1/retrain
@app.route("/api/v1/retrain/", methods=["GET"])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"
'''
if __name__ == '__main__':
    app.run(debug=True)
