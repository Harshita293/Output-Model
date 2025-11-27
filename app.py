# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import geopy.distance
import requests

app = Flask(__name__)

# --- 1. LOAD ALL MODELS AND THE DATASET ---
try:
    model_suitability = joblib.load('well_success_ensemble_model.pkl')
    model_depth = joblib.load('well_depth_pipeline.pkl')
    model_yield = joblib.load('hybrid_well_yield_model.pkl')
    model_drilling = joblib.load('drilling_method_ensemble_pipeline.pkl')
    model_soil_type = joblib.load('soil_type_model.pkl')  # NEW MODEL

    print("All models loaded successfully!")

except Exception as e:
    print(f"CRITICAL ERROR: Model file missing -> {e}")
    model_suitability = model_depth = model_yield = model_drilling = model_soil_type = None

try:
    dataset = pd.read_csv('india_water_well_dataset.csv')

    if 'yield_lpm_log' not in dataset.columns and 'yield_lpm' in dataset.columns:
        dataset['yield_lpm_log'] = np.log1p(dataset['yield_lpm'])

    print("Dataset loaded successfully!")

    SUITABILITY_FEATURES = [
        'latitude', 'longitude', 'elevation_m', 'slope_deg', 'avg_rainfall_mm', 
        'rainy_days', 'distance_to_river_km', 'depth_to_bedrock_m', 'ndvi_mean', 
        'ndvi_dry', 'ndvi_wet', 'soil_type', 'landcover', 'drilling_method', 
        'well_depth_m', 'screen_length_m', 'diameter_in'
    ]
    
    encoders = {}
    categorical_columns = ['soil_type', 'landcover', 'drilling_method']
    for col in categorical_columns:
        dataset[col] = dataset[col].astype(str)
        le = LabelEncoder()
        
        all_labels = list(dataset[col].unique())
        if col == 'drilling_method':
            for extra in ['down_the_hole', 'manual', 'rotary']:
                if extra not in all_labels:
                    all_labels.append(extra)
        
        le.fit(all_labels)
        encoders[col] = le
    print("Encoders prepared.")

except Exception as e:
    print(f"CRITICAL ERROR loading dataset: {e}")
    dataset = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    lat = float(request.form.get("lat"))
    lng = float(request.form.get("lng"))
    
    closest_data_df = find_closest_data_point(lat, lng)
    prediction_results = generate_predictions_from_data(closest_data_df)

    return render_template("dashboard.html", results=prediction_results, lat=lat, lng=lng)


@app.route("/find_nearby", methods=["POST"])
def find_nearby():
    original_lat = float(request.form.get("original_lat"))
    original_lng = float(request.form.get("original_lng"))

    final_results = None

    nearest_50 = find_n_closest_points(original_lat, original_lng, 50)
    if nearest_50 is not None:
        for index, row in nearest_50.iterrows():
            neighbor_df = pd.DataFrame([row])

            if predict_suitability(neighbor_df) == 1:
                final_results = generate_predictions_from_data(neighbor_df)

                found_coords = (row["latitude"], row["longitude"])
                distance = round(
                    geopy.distance.geodesic((original_lat, original_lng), found_coords).km, 2
                )
                display_name = get_location_name(found_coords[0], found_coords[1])
                try:
                    soil_type_pred=model_soil_type.predict(neighbor_df)[0]
                except:
                    soil_type_pred = neighbor_df["soil_type"].iloc[0]

                # ADD SOIL TYPE TO NEARBY LOCATION
                final_results["nearby_suitable_location"] = {
                    "lat": found_coords[0],
                    "lng": found_coords[1],
                    "distance_km": distance,
                    "display_name": display_name,
                    "soil_type": soil_type_pred
                    # "soil_type": final_results["soil_type"]
                }
                break

    if final_results is None:
        final_results = generate_predictions_from_data(
            find_closest_data_point(original_lat, original_lng)
        )
        final_results["nearby_suitable_location"] = {"not_found": True}

    return render_template(
        "dashboard.html",
        results=final_results,
        lat=original_lat,
        lng=original_lng,
    )


# ----------------------------------------------
#      PREDICTION LOGIC FUNCTION
# ----------------------------------------------
def generate_predictions_from_data(df):
    suitability = "Unavailable"
    depth = yield_lph = drilling_method = soil_type_pred= "N/A"
    # model_soil_type = "N/A"

    if df is not None and all(
        [model_suitability, model_depth, model_yield, model_drilling, model_soil_type]
    ):
        # Predict Suitability
        prediction_suitability = predict_suitability(df)
        suitability = "Suitable" if prediction_suitability == 1 else "Not Suitable"


        if suitability == "Suitable":
            features = df.copy()
            features["success"] = prediction_suitability

            # Depth prediction
            prediction_depth = model_depth.predict(features)[0]
            depth = f"{round(prediction_depth, 1)} m"

            # Yield prediction
            features_yield = features.copy()
            features_yield["depth_ratio"] = features_yield["well_depth_m"] / features_yield["depth_to_bedrock_m"]
            features_yield["ndvi_range"] = features_yield["ndvi_wet"] - features_yield["ndvi_dry"]
            features_yield["rainfall_per_day"] = features_yield["avg_rainfall_mm"] / features_yield["rainy_days"]
            features_yield.replace([np.inf, -np.inf], 0, inplace=True)

            prediction_yield = model_yield.predict(features_yield)[0]
            yield_lph = f"~{round(prediction_yield)} LPH"

            # Drilling method
            features_drilling = features.copy()
            features_drilling["rain_per_day"] = features_drilling["avg_rainfall_mm"] / features_drilling["rainy_days"]
            features_drilling["elevation_minus_bedrock"] = (
                features_drilling["elevation_m"] - features_drilling["depth_to_bedrock_m"]
            )
            features_drilling.replace([np.inf, -np.inf], 0, inplace=True)

            prediction_drilling = model_drilling.predict(features_drilling)[0]
            if isinstance(prediction_drilling, str):
                drilling_method = prediction_drilling
            else:
                drilling_method = encoders["drilling_method"].inverse_transform(
                    [prediction_drilling]
                )[0]

        else:
            depth = yield_lph = drilling_method = "Not Applicable"

    return {
        "suitability": suitability,
        "soil_type": soil_type_pred,        # NEW ✔
        "expected_depth_m": depth,
        "yield_lph": yield_lph,
        "drilling_method": drilling_method,
    }


def predict_suitability(df):
    features = df[SUITABILITY_FEATURES].copy()
    for col, encoder in encoders.items():
        features[col] = transform_with_encoder(features[col], encoder)
    return model_suitability.predict(features)[0]


def transform_with_encoder(data, encoder):
    unseen_labels = set(data) - set(encoder.classes_)
    if unseen_labels:
        data = data.replace(list(unseen_labels), encoder.classes_[0])
    return encoder.transform(data)


def find_closest_data_point(lat, lng):
    if dataset is not None:
        distances = (dataset["latitude"] - lat) ** 2 + (dataset["longitude"] - lng) ** 2
        return dataset.loc[[distances.idxmin()]]
    return None


def find_n_closest_points(lat, lng, n):
    if dataset is not None:
        distances = (dataset["latitude"] - lat) ** 2 + (dataset["longitude"] - lng) ** 2
        return dataset.loc[distances.nsmallest(n).index]
    return None


def get_location_name(lat, lng):
    try:
        api_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&accept-language=en"
        response = requests.get(api_url, headers={"User-Agent": "AquaSightApp/1.0"})
        response.raise_for_status()
        return response.json().get("display_name", "Address Not Found")
    except:
        return "Address Not Found"


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
