# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import weatherkey
import pickle
import io


crop_recommendation_model_path = 'RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


def fetch(city_name):
    api_key = weatherkey.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    a = response.json()
    if a["cod"] != "404":
        b = a["main"]

        temperature = round((b["temp"] - 273.15), 2)
        humidity = b["humidity"]
        return temperature, humidity
    else:
        return None


app = Flask(__name__)


@ app.route('/')
def home():
    title = 'Home'
    return render_template('main.html', title=title)


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'FamTech - Crop Predictor'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        state = request.form.get("stt")
        city = request.form.get("city")

        if fetch(city) != None:
            temperature, humidity = fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('result.html', prediction=final_prediction, title=title)

        else:

            return render_template('retry.html', title=title)


if __name__ == '__main__':
    app.run(debug=True)
