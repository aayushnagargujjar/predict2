# app.py

from flask import Flask, request, jsonify
from prophet import Prophet
import pandas as pd
import traceback
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_forecast():
    try:
        data = request.get_json()
        co2_data = data.get('co2_data', [])
        water_data = data.get('water_data', [])

        if not co2_data or len(co2_data) < 3:
            return jsonify({"error": "Not enough CO₂ data"}), 400

        df_co2 = pd.DataFrame(co2_data).rename(columns={"date": "ds", "value": "y"})
        df_co2['ds'] = pd.to_datetime(df_co2['ds'])
        model_co2 = Prophet()
        model_co2.fit(df_co2)
        future_co2 = model_co2.make_future_dataframe(periods=3)
        forecast_co2 = model_co2.predict(future_co2)[['ds', 'yhat']].tail(3)
        co2_forecast = [{"date": str(row['ds'].date()), "co2_pred": round(row['yhat'], 2)} for _, row in forecast_co2.iterrows()]

        water_forecast = []
        if water_data and len(water_data) >= 3:
            df_water = pd.DataFrame(water_data).rename(columns={"date": "ds", "value": "y"})
            df_water['ds'] = pd.to_datetime(df_water['ds'])
            model_water = Prophet()
            model_water.fit(df_water)
            future_water = model_water.make_future_dataframe(periods=3)
            forecast_water = model_water.predict(future_water)[['ds', 'yhat']].tail(3)
            water_forecast = [{"date": str(row['ds'].date()), "water_pred": round(row['yhat'], 2)} for _, row in forecast_water.iterrows()]

        combined = []
        for co2_item in co2_forecast:
            item = co2_item.copy()
            match = next((w for w in water_forecast if w['date'] == co2_item['date']), None)
            item['water_pred'] = match['water_pred'] if match else 0.0
            combined.append(item)

        return jsonify({"forecast": combined}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
