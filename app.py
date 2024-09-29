from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load your model, PCA transformer, and scaler
model = joblib.load("models/binaryclassrandomsearchmodelrf.joblib")
pca = joblib.load("models/pca_transformer.joblib")
scaler = joblib.load("models/scaler.joblib")

# Function to preprocess the input data
def preprocess_flight_data(date_str, std_str, sta_str, from_city, to_city):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    std = datetime.strptime(std_str, "%H:%M")
    sta = datetime.strptime(sta_str, "%H:%M")

    weekday = date.isoweekday()
    std_hour = round(std.hour + std.minute / 60, 2)
    sta_hour = round(sta.hour + sta.minute / 60, 2)

    # Initialize a dictionary with all cities set to False
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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_message = ""
    
    if request.method == "POST":
        date_str = request.form["date"]
        std_str = request.form["std"]
        sta_str = request.form["sta"]
        from_city = request.form["from_city"]
        to_city = request.form["to_city"]

        # Check for same airport
        if from_city == to_city:
            prediction_message = "Error: Departure and arrival airports cannot be the same."
            return render_template('index.html', prediction_message=prediction_message, prediction=prediction)

        # Check for same time
        if std_str == sta_str:
            prediction_message = "Error: Scheduled departure time and scheduled arrival time cannot be the same."
            return render_template('index.html', prediction_message=prediction_message, prediction=prediction)

        # Check if the date is today or in the future
        today = datetime.today().date()
        input_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        if input_date < today:
            prediction_message = "Error: The chosen date cannot be in the past."
            return render_template('index.html', prediction_message=prediction_message, prediction=prediction)

        std_time = datetime.strptime(std_str, "%H:%M")
        sta_time = datetime.strptime(sta_str, "%H:%M")
        # Check if flight time is more than 20 minutes using absolute difference
        if abs((sta_time - std_time).total_seconds()) < 60 * 60:  # 20 minutes in seconds
            prediction_message = "Error: Flight time is incorrect Please use correct time"
            return render_template('index.html', prediction_message=prediction_message, prediction=prediction)

        # Process the flight data
        flight_df = preprocess_flight_data(date_str, std_str, sta_str, from_city, to_city)
        scaled_data = scaler.transform(flight_df)
        input_data_pca = pca.transform(scaled_data)
        prediction = model.predict(input_data_pca)[0]  
        
        if prediction == 0:
            prediction_message = (
                f"The flights on {date_str} "
                f"at {std_str} from <span class='highlight'>{from_city}</span> "
                f"to <span class='highlight'>{to_city}</span> are <span class='highlight'>On time.</span>"
            )
        elif prediction == 1:
            prediction_message = (
                f"The flights from <span class='highlight'>{from_city}</span> to <span class='highlight'>{to_city}</span> "
                f"could be <span class='highlight'>delayed</span> on {date_str} at {std_str}."
            )

    return render_template('index.html', prediction_message=prediction_message, prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)