from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
from werkzeug.exceptions import BadRequest

app = Flask(__name__)

# Load your model, PCA transformer, and scaler
model = joblib.load("models/binaryclassrandomsearchmodelrf.joblib")
pca = joblib.load("models/pca_transformer.joblib")
scaler = joblib.load("models/scaler.joblib")

def preprocess_flight_data(date_str, std_str, sta_str, from_city, to_city):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    std = datetime.strptime(std_str, "%H:%M")
    sta = datetime.strptime(sta_str, "%H:%M")

    weekday = date.isoweekday()
    std_hour = round(std.hour + std.minute / 60, 2)
    sta_hour = round(sta.hour + sta.minute / 60, 2)

    all_cities = {
        "From__AMD": False, "From__ATQ": False, "From__BBI": False, "From__BDQ": False,
        "From__BHO": False, "From__BLR": True, "From__BOM": False, "From__CCJ": False,
        "From__CCU": False, "From__CJB": False, "From__COK": False, "From__DEL": False,
        "From__GAU": False, "From__GOI": False, "From__GOX": False, "From__HYD": False,
        "From__IDR": False, "From__IXA": False, "From__IXC": False, "From__IXJ": False,
        "From__IXL": False, "From__IXR": False, "From__JAI": False, "From__JDH": False,
        "From__LKO": False, "From__MAA": False, "From__PAT": True, "From__PNQ": False,
        "From__RAJ": False, "From__RDP": False, "From__SXR": False, "From__TLS": False,
        "From__TRV": False, "From__UDR": False, "To__AGR": False, "To__AMD": False,
        "To__BBI": False, "To__BDQ": False, "To__BHO": False, "To__BLR": False, "To__BOM": False,
        "To__CCU": False, "To__CJB": False, "To__COK": False, "To__DEL": False, "To__GAY": False,
        "To__GOI": False, "To__GOX": False, "To__HYD": False, "To__IDR": False, "To__IXA": False,
        "To__IXC": False, "To__IXL": False, "To__IXM": False, "To__IXR": False, "To__IXS": False,
        "To__JAI": False, "To__LKO": False, "To__MAA": False, "To__NAG": False, "To__PAT": False,
        "To__PNQ": False, "To__RAJ": False, "To__RDP": False, "To__STV": False, "To__SXR": False,
        "To__UDR": False
    }

    all_cities[f"From__{from_city}"] = True
    all_cities[f"To__{to_city}"] = True

    flight_data = {"Day": weekday, "STD": std_hour, "STA": sta_hour, **all_cities}

    return pd.DataFrame([flight_data])

@app.route("/predict", methods=["POST"])
def predict():
    prediction = None
    status = None
    try:
        data = request.get_json()

        # Validate input data
        date_str = data.get("date")
        std_str = data.get("std")
        sta_str = data.get("sta")
        from_city = data.get("from_city")
        to_city = data.get("to_city")

        # Check for missing fields
        if not all([date_str, std_str, sta_str, from_city, to_city]):
            raise BadRequest("Missing required fields.")

        # Validate date format
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise BadRequest("Invalid date format. Use YYYY-MM-DD.")

        # Validate time format
        for time_str in [std_str, sta_str]:
            try:
                datetime.strptime(time_str, "%H:%M")
            except ValueError:
                raise BadRequest(f"Invalid time format: {time_str}. Use HH:MM.")

        # Validate city names
        valid_cities = ["AMD", "ATQ", "BBI", "BDQ", "BHO", "BLR", "BOM", "CCJ", "CCU", 
                        "CJB", "COK", "DEL", "GAU", "GOI", "GOX", "HYD", "IDR", 
                        "IXA", "IXC", "IXJ", "IXL", "IXR", "JAI", "JDH", "LKO", 
                        "MAA", "PAT", "PNQ", "RAJ", "RDP", "SXR", "TLS", "TRV", 
                        "UDR", "AGR", "GAY", "NAG", "IXM", "IXS", "STV"]
        
        if from_city not in valid_cities or to_city not in valid_cities:
            raise BadRequest(f"Invalid city names: {from_city}, {to_city}.")

        # Process flight data
        flight_df = preprocess_flight_data(date_str, std_str, sta_str, from_city, to_city)
        scaled_data = scaler.transform(flight_df)
        input_data_pca = pca.transform(scaled_data)
        prediction = model.predict(input_data_pca)[0]

        if prediction == 0:
            status = "on-time"
        elif prediction == 1:
            status = "delayed"

        return jsonify({
            "status": status,
        })

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred while processing the request."}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
