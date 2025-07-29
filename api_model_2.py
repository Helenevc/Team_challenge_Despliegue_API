from flask import Flask, jsonify, request
import pickle
import numpy as np
app = Flask(__name__)
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API de predicciones de adicción a las redes sociales"
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    # Cargar el modelo
    try:
        with open('model_2.pkl', 'rb') as f:
            model_2 = pickle.load(f)
    except Exception as e:
        return jsonify({"error": f"No se pudo cargar el modelo: {str(e)}"}), 500
    # Obtener parámetros
    age = request.args.get('Age', None)
    continent = request.args.get('Continent', None)
    sleep = request.args.get('Sleep_Hours_Per_Night', None)
    # Validar existencia
    if age is None or continent is None or sleep is None:
        return jsonify({"error": "Faltan argumentos. Se requieren Age, Continent y Sleep_Hours_Per_Night."}), 400
    # One-hot encoding de Age
    col_age = ["Age_19", "Age_20", "Age_21", "Age_22", "Age_23", "Age_24"]
    age_dummies = {col: 0 for col in col_age}
    selected_col_age = f"Age_{age}"
    if selected_col_age in age_dummies:
        age_dummies[selected_col_age] = 1
    else:
        return jsonify({"error": f"Valor de edad no permitido: '{age}'"}), 400
    # One-hot encoding de Continent
    col_cont = ["Continent_Asia", "Continent_Europe", "Continent_North America", "Continent_Oceania", "Continent_South America"]
    continent_dummies = {col: 0 for col in col_cont}
    selected_col_cont = f"Continent_{continent}"
    if selected_col_cont in continent_dummies:
        continent_dummies[selected_col_cont] = 1
    else:
        return jsonify({"error": f"Continente no reconocido: '{continent}'"}), 400
    try:
        input_vector = [float(sleep)] + list(continent_dummies.values()) + list(age_dummies.values())
        prediction = model_2.predict([input_vector])
        prediction_value = float(prediction[0])  # Conversión segura a tipo JSON serializable
    except Exception as e:
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500
    return jsonify({'predictions': prediction_value})
if __name__ == '__main__':
    app.run(debug=True)