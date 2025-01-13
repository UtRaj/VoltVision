from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load trained model
model = load_model("model/lstm_model.h5")

# Load dataset and preprocess
df = pd.read_csv("data/events.csv")
df.rename(columns={"End time UTC+03:00": "DateTime", "Electricity consumption in Finland": "Consumption"}, inplace=True)
df = df.set_index("DateTime")
df.index = pd.to_datetime(df.index)

# Aggregate data to daily level
y = df[["Consumption"]].resample("D").mean()

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y)

def forecast_future_days(n_days, last_days, model, scaler):
    temp_input = last_days.flatten().tolist()
    lst_output = []
    n_steps = len(last_days)

    for _ in range(n_days):
        x_input = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(float(yhat[0][0]))
        lst_output.append(yhat[0][0])

    return scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        past_days = int(request.form["past_days"])
        future_days = int(request.form["future_days"])
        
        last_days = y_scaled[-past_days:]
        forecast = forecast_future_days(future_days, last_days, model, scaler)
        
        forecast_dates = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=future_days).strftime("%Y-%m-%d").tolist()
        forecast_values = forecast.flatten().tolist()

        # Calculate additional statistics
        forecast_mean = np.mean(forecast_values)
        forecast_min = np.min(forecast_values)
        forecast_max = np.max(forecast_values)
        forecast_std = np.std(forecast_values)
        forecast_median = np.median(forecast_values)
        forecast_range = forecast_max - forecast_min

        # Calculate the overall mean of the consumption
        overall_mean = np.mean(y)
        
        # Determine the direction of the forecast (increasing or decreasing)
        trend_direction = "increasing" if forecast_values[-1] > forecast_values[0] else "decreasing"
        
        # Calculate percentage change in forecasted values
        percent_change = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y.index[-past_days:], scaler.inverse_transform(last_days), label=f'Past {past_days} Days')
        plt.plot(pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=future_days),
                 forecast, label=f'Forecast for {future_days} Days', linestyle='dashed', color='red')
        plt.xlabel("Date")
        plt.ylabel("Electricity Consumption (MWh)")
        plt.legend()
        plt.title("Electricity Consumption Forecast")

        # Save plot as image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Zip forecast dates and values
        forecast_zipped = zip(forecast_dates, forecast_values)

        return render_template("results.html", past_days=past_days, future_days=future_days,
                               forecast_zipped=forecast_zipped, plot_url=plot_url,
                               forecast_mean=forecast_mean, forecast_min=forecast_min,
                               forecast_max=forecast_max, forecast_std=forecast_std,
                               forecast_median=forecast_median, forecast_range=forecast_range,
                               overall_mean=overall_mean, trend_direction=trend_direction,
                               percent_change=percent_change)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
