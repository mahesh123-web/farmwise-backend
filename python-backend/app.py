"""
FarmWise Advisor — Python Flask Backend
Uses your trained ML models directly — no Gemini needed for core predictions!

Models used:
  soil_vision_model.pth  — EfficientNet-B2 (PyTorch) — soil type from image
  crop_recommender.pkl   — RandomForest (Sklearn)     — crop recommendation
  label_encoder.pkl      — LabelEncoder               — crop index → name
  scaler.pkl             — StandardScaler             — feature normalization

Routes:
  POST /api/analyze-soil        — soil image → soil type + N,P,K,pH
  POST /api/recommendations     — soil + weather → top 3 crops + prices
  GET  /api/weather             — Open-Meteo 7-day forecast (free)
  GET  /api/geocode             — city name → lat/lon (free)
  GET  /api/markets             — prices for all crops
  GET  /api/health              — health check
"""

import os, io, json
import numpy as np
import pandas as pd
import joblib
import torch
import timm
import requests
from PIL import Image
from torchvision import transforms
from flask import Flask, request, jsonify

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── CORS: handle every request including OPTIONS preflight ─────────────────────
@app.before_request
def handle_preflight():
    from flask import request as req
    if req.method == "OPTIONS":
        from flask import make_response
        res = make_response()
        res.headers["Access-Control-Allow-Origin"]  = "*"
        res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        res.headers["Access-Control-Max-Age"]       = "3600"
        return res, 200

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Alluvial', 'Black', 'Cinder', 'Clay', 'Laterite', 'Peat', 'Red', 'Yellow']

# ─── Soil Properties Lookup ───────────────────────────────────────────────────
# Maps soil type → estimated N, P, K, pH, moisture
SOIL_PROPERTIES = {
    "Alluvial": {"N": 60, "P": 45, "K": 40, "ph": 7.0, "moisture": "High"},
    "Black":    {"N": 50, "P": 35, "K": 50, "ph": 7.5, "moisture": "Very High"},
    "Cinder":   {"N": 15, "P": 10, "K": 8,  "ph": 6.0, "moisture": "Very Low"},
    "Clay":     {"N": 45, "P": 30, "K": 25, "ph": 7.2, "moisture": "High"},
    "Laterite": {"N": 20, "P": 15, "K": 10, "ph": 5.5, "moisture": "Low"},
    "Peat":     {"N": 35, "P": 20, "K": 15, "ph": 4.5, "moisture": "Very High"},
    "Red":      {"N": 25, "P": 20, "K": 15, "ph": 6.2, "moisture": "Low"},
    "Yellow":   {"N": 30, "P": 22, "K": 18, "ph": 6.0, "moisture": "Medium"},
}

# ─── Market Prices ────────────────────────────────────────────────────────────
MARKET_PRICES = {
    "rice":        {"price": 2200,  "trend": "up"},
    "wheat":       {"price": 2275,  "trend": "stable"},
    "maize":       {"price": 1850,  "trend": "up"},
    "cotton":      {"price": 6800,  "trend": "down"},
    "jute":        {"price": 5000,  "trend": "stable"},
    "coconut":     {"price": 3200,  "trend": "up"},
    "coffee":      {"price": 18000, "trend": "up"},
    "banana":      {"price": 1200,  "trend": "stable"},
    "mango":       {"price": 4500,  "trend": "up"},
    "grapes":      {"price": 7500,  "trend": "stable"},
    "apple":       {"price": 9000,  "trend": "up"},
    "orange":      {"price": 3500,  "trend": "stable"},
    "papaya":      {"price": 1500,  "trend": "up"},
    "muskmelon":   {"price": 800,   "trend": "stable"},
    "watermelon":  {"price": 600,   "trend": "down"},
    "pomegranate": {"price": 8000,  "trend": "up"},
    "lentil":      {"price": 5500,  "trend": "stable"},
    "chickpea":    {"price": 5200,  "trend": "up"},
    "kidneybeans": {"price": 6000,  "trend": "stable"},
    "blackgram":   {"price": 4800,  "trend": "up"},
    "mungbean":    {"price": 7000,  "trend": "stable"},
    "mothbeans":   {"price": 4500,  "trend": "stable"},
    "pigeonpeas":  {"price": 6200,  "trend": "up"},
}

# Crop growing info
CROP_INFO = {
    "rice":        {"duration": "120-130 days", "water": "High",      "season": "Kharif"},
    "wheat":       {"duration": "110-120 days", "water": "Medium",    "season": "Rabi"},
    "maize":       {"duration": "80-90 days",   "water": "Medium",    "season": "Kharif"},
    "cotton":      {"duration": "150-180 days", "water": "Medium",    "season": "Kharif"},
    "jute":        {"duration": "100-120 days", "water": "High",      "season": "Kharif"},
    "coconut":     {"duration": "5-6 years",    "water": "High",      "season": "Perennial"},
    "coffee":      {"duration": "3-4 years",    "water": "Medium",    "season": "Perennial"},
    "banana":      {"duration": "10-12 months", "water": "High",      "season": "Perennial"},
    "mango":       {"duration": "5-8 years",    "water": "Low",       "season": "Perennial"},
    "grapes":      {"duration": "2-3 years",    "water": "Medium",    "season": "Perennial"},
    "apple":       {"duration": "5-8 years",    "water": "Medium",    "season": "Perennial"},
    "orange":      {"duration": "3-5 years",    "water": "Medium",    "season": "Perennial"},
    "papaya":      {"duration": "9-10 months",  "water": "Medium",    "season": "Perennial"},
    "muskmelon":   {"duration": "80-90 days",   "water": "Low",       "season": "Summer"},
    "watermelon":  {"duration": "80-90 days",   "water": "Low",       "season": "Summer"},
    "pomegranate": {"duration": "2-3 years",    "water": "Low",       "season": "Perennial"},
    "lentil":      {"duration": "100-120 days", "water": "Low",       "season": "Rabi"},
    "chickpea":    {"duration": "90-95 days",   "water": "Low",       "season": "Rabi"},
    "kidneybeans": {"duration": "90-120 days",  "water": "Medium",    "season": "Kharif"},
    "blackgram":   {"duration": "60-65 days",   "water": "Low",       "season": "Kharif"},
    "mungbean":    {"duration": "60-65 days",   "water": "Low",       "season": "Kharif"},
    "mothbeans":   {"duration": "75-90 days",   "water": "Very Low",  "season": "Kharif"},
    "pigeonpeas":  {"duration": "130-160 days", "water": "Low",       "season": "Kharif"},
}

# ─── Load Models ──────────────────────────────────────────────────────────────

print("Loading models...")

# Vision model — EfficientNet-B2
vision_model = timm.create_model(
    'efficientnet_b2',
    pretrained=False,
    num_classes=len(CLASS_NAMES)
)
vision_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "soil_vision_model.pth"), map_location=DEVICE)
)
vision_model = vision_model.to(DEVICE)
vision_model.eval()
print(f"✅ Vision model loaded on {DEVICE}")

# Crop recommendation model
crop_model = joblib.load(os.path.join(MODEL_DIR, "crop_recommender.pkl"))
le         = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
scaler     = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print("✅ Crop recommendation model loaded")

# Image preprocessing (same as training)
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print("✅ All models ready!\n")

# ─── Helper Functions ─────────────────────────────────────────────────────────

def analyze_soil_image(image_bytes):
    """Run soil image through EfficientNet-B2 → soil type + properties"""
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = img_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output     = vision_model(tensor)
        probs      = torch.softmax(output, dim=1)
        confidence = probs.max().item()
        class_idx  = probs.argmax().item()

    soil_type  = CLASS_NAMES[class_idx]
    properties = SOIL_PROPERTIES[soil_type].copy()

    return {
        "soilType":      soil_type,
        "confidence":    round(confidence * 100, 2),
        "ph":            properties["ph"],
        "phCategory":    ph_category(properties["ph"]),
        "nitrogen":      f"{properties['N']} kg/ha",
        "phosphorus":    f"{properties['P']} kg/ha",
        "potassium":     f"{properties['K']} kg/ha",
        "organicMatter": organic_matter(soil_type),
        "moisture":      properties["moisture"],
        "texture":       soil_texture(soil_type),
        "drainage":      soil_drainage(soil_type),
        "erosionRisk":   erosion_risk(soil_type),
        "recommendations": soil_recommendations(soil_type, properties),
        "_raw": properties,  # keep raw for crop model
    }

def get_crop_recommendations(soil_props, weather, top_n=3):
    """Run crop recommender RandomForest → top N crops with confidence"""
    features = pd.DataFrame([[
        soil_props["N"],
        soil_props["P"],
        soil_props["K"],
        weather.get("temperature", 25),
        weather.get("humidity", 70),
        soil_props["ph"],
        weather.get("rainfall", 100),
    ]], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

    features_scaled = scaler.transform(features)
    probs   = crop_model.predict_proba(features_scaled)[0]
    top_idx = probs.argsort()[-top_n:][::-1]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        crop_name  = le.classes_[idx]
        confidence = round(float(probs[idx]) * 100, 1)
        price_data = MARKET_PRICES.get(crop_name.lower(), {"price": 0, "trend": "stable"})
        info       = CROP_INFO.get(crop_name.lower(), {"duration": "N/A", "water": "Medium", "season": "N/A"})

        results.append({
            "rank":               rank,
            "name":               crop_name.capitalize(),
            "suitabilityScore":   confidence,
            "bestPricePerQuintal": price_data["price"],
            "avgPrice":           price_data["price"],
            "trend":              price_data["trend"],
            "bestMarket":         "City Market",
            "estimatedProfitPerAcre": estimate_profit(crop_name, price_data["price"]),
            "growthDuration":     info["duration"],
            "waterRequirement":   info["water"],
            "season":             info["season"],
            "soilFitReasons":     crop_soil_reasons(crop_name, soil_props),
            "marketFitReasons":   [f"Price ₹{price_data['price']}/quintal — {price_data['trend']} trend"],
            "weatherFitReasons":  crop_weather_reasons(crop_name, weather),
            "requirements":       f"{info['water']} water, {info['season']} season crop",
            "riskFactors":        crop_risks(crop_name),
            "mitigationTips":     crop_tips(crop_name),
            "organicFeasible":    True,
            "governmentSchemes":  ["PM-KISAN", "PMFBY", "e-NAM"],
        })

    return results

# ─── Utility Functions ────────────────────────────────────────────────────────

def ph_category(ph):
    if ph < 5.5:   return "Acidic"
    if ph < 6.5:   return "Slightly Acidic"
    if ph < 7.5:   return "Neutral"
    if ph < 8.5:   return "Slightly Alkaline"
    return "Alkaline"

def organic_matter(soil_type):
    om = {"Alluvial": "3.5%", "Black": "4.2%", "Cinder": "0.8%", "Clay": "3.0%",
          "Laterite": "1.5%", "Peat": "8.0%", "Red": "1.8%", "Yellow": "2.0%"}
    return om.get(soil_type, "2.0%")

def soil_texture(soil_type):
    tx = {
        "Alluvial": "Fine to medium texture, well-drained, high fertility",
        "Black":    "Heavy clay texture, high water retention, cracks when dry",
        "Cinder":   "Coarse volcanic texture, very low water retention",
        "Clay":     "Fine texture, sticky when wet, poor drainage",
        "Laterite": "Coarse texture, hardened surface, leached of nutrients",
        "Peat":     "Spongy organic texture, very high water retention",
        "Red":      "Sandy loam texture, low water retention, well-drained",
        "Yellow":   "Medium texture, moderate drainage, moderate fertility",
    }
    return tx.get(soil_type, "Medium texture")

def soil_drainage(soil_type):
    dr = {"Alluvial": "Good", "Black": "Poor", "Cinder": "Excellent",
          "Clay": "Poor", "Laterite": "Moderate", "Peat": "Poor",
          "Red": "Good", "Yellow": "Moderate"}
    return dr.get(soil_type, "Moderate")

def erosion_risk(soil_type):
    er = {"Alluvial": "Medium", "Black": "Low", "Cinder": "High",
          "Clay": "Low", "Laterite": "High", "Peat": "Medium",
          "Red": "High", "Yellow": "Medium"}
    return er.get(soil_type, "Medium")

def soil_recommendations(soil_type, props):
    recs = [f"pH {props['ph']} — {ph_category(props['ph'])} soil, suitable for most crops"]
    if props["N"] < 30:
        recs.append("Low nitrogen — apply urea or compost before sowing")
    elif props["N"] > 55:
        recs.append("High nitrogen — reduce N fertilizer to avoid over-growth")
    else:
        recs.append("Adequate nitrogen levels for most crops")
    if props["P"] < 20:
        recs.append("Low phosphorus — apply DAP or SSP fertilizer")
    else:
        recs.append("Good phosphorus levels support root and flower development")
    if props["moisture"] in ["Low", "Very Low"]:
        recs.append("Low moisture — irrigation will be essential for good yield")
    elif props["moisture"] in ["High", "Very High"]:
        recs.append("High moisture — ensure proper drainage to prevent root rot")
    return recs

def estimate_profit(crop_name, price):
    # Rough estimate based on typical Indian yields
    yields = {"rice": 25, "wheat": 20, "maize": 22, "cotton": 8, "mungbean": 7,
              "pomegranate": 40, "orange": 50, "banana": 120, "sugarcane": 300}
    yield_q  = yields.get(crop_name.lower(), 15)
    cost_per_acre = 15000
    return max(0, (yield_q * price) - cost_per_acre)

def crop_soil_reasons(crop_name, soil_props):
    reasons = [
        f"Soil pH {soil_props['ph']} suits {crop_name} growth requirements",
        f"N:{soil_props['N']} P:{soil_props['P']} K:{soil_props['K']} kg/ha available",
    ]
    return reasons

def crop_weather_reasons(crop_name, weather):
    temp = weather.get("temperature", 25)
    rain = weather.get("rainfall", 100)
    return [
        f"Temperature {temp}°C is suitable for {crop_name}",
        f"Expected rainfall {rain}mm matches water needs",
    ]

def crop_risks(crop_name):
    risks = {
        "rice":        ["Water logging during heavy monsoon", "Blast disease risk"],
        "wheat":       ["Late frost risk", "Rust disease susceptibility"],
        "maize":       ["Fall armyworm pest risk", "Drought sensitivity at tasseling"],
        "cotton":      ["Bollworm infestation", "Requires long frost-free season"],
        "pomegranate": ["Fruit borer pest", "Aril cracking in humid conditions"],
        "mungbean":    ["Powdery mildew risk", "Yellow mosaic virus susceptibility"],
        "orange":      ["Citrus canker disease", "Frost damage to young plants"],
    }
    return risks.get(crop_name.lower(), ["Monitor for common pests", "Ensure adequate irrigation"])

def crop_tips(crop_name):
    tips = {
        "rice":        ["Use SRI method for 20-30% higher yield", "Apply zinc sulphate at transplanting"],
        "wheat":       ["Use certified seeds for better germination", "Apply irrigation at crown root stage"],
        "cotton":      ["Use BT cotton for bollworm resistance", "Monitor for pink bollworm after 60 days"],
        "pomegranate": ["Use drip irrigation for water efficiency", "Apply Bordeaux mixture for disease control"],
        "mungbean":    ["Inoculate seeds with Rhizobium before sowing", "Harvest when 80% pods turn brown"],
    }
    return tips.get(crop_name.lower(), ["Follow recommended fertilizer schedule", "Regular field monitoring recommended"])

def wmo_label(code):
    if code == 0:   return {"label": "Clear Sky",     "emoji": "☀️"}
    if code <= 2:   return {"label": "Partly Cloudy", "emoji": "⛅"}
    if code == 3:   return {"label": "Overcast",      "emoji": "☁️"}
    if code <= 49:  return {"label": "Foggy",         "emoji": "🌫️"}
    if code <= 59:  return {"label": "Drizzle",       "emoji": "🌦️"}
    if code <= 69:  return {"label": "Rain",          "emoji": "🌧️"}
    if code <= 79:  return {"label": "Snow",          "emoji": "❄️"}
    if code <= 82:  return {"label": "Rain Showers",  "emoji": "🌧️"}
    if code <= 99:  return {"label": "Thunderstorm",  "emoji": "⛈️"}
    return {"label": "Unknown", "emoji": "🌡️"}

# ─── Routes ───────────────────────────────────────────────────────────────────


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models": "loaded", "device": str(DEVICE)})


@app.route("/api/analyze-soil", methods=["POST", "OPTIONS"])
def analyze_soil():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file  = request.files["image"]
        image_bytes = image_file.read()
        location    = request.form.get("location", "unspecified")

        result = analyze_soil_image(image_bytes)
        # Remove internal raw field before sending
        result.pop("_raw", None)

        return jsonify({"success": True, "analysis": result})

    except Exception as e:
        print(f"[analyze-soil error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations", methods=["POST", "OPTIONS"])
def recommendations():
    try:
        data         = request.get_json()
        soil_analysis = data.get("soilAnalysis", {})
        weather_data  = data.get("weatherData", {})
        location      = data.get("location", "India")

        # Extract raw N,P,K,ph from soil analysis
        # The soil_analysis from /api/analyze-soil has strings like "25 kg/ha"
        # so we parse them back to numbers
        def parse_num(val):
            if isinstance(val, (int, float)): return val
            return float(str(val).split()[0])

        soil_props = {
            "N":  parse_num(soil_analysis.get("nitrogen",   25)),
            "P":  parse_num(soil_analysis.get("phosphorus", 20)),
            "K":  parse_num(soil_analysis.get("potassium",  15)),
            "ph": parse_num(soil_analysis.get("ph",         6.5)),
        }

        # Extract weather from nested structure
        weather_current = weather_data.get("weather", weather_data)
        if "current" in weather_current:
            weather_current = weather_current["current"]

        weather = {
            "temperature": weather_current.get("temp", 25),
            "humidity":    weather_current.get("humidity", 70),
            "rainfall":    100,  # default; update if you have forecast rainfall sum
        }

        # Also try to get rainfall from forecast
        forecast = data.get("weatherData", {}).get("weather", {}).get("forecast", [])
        if forecast:
            weather["rainfall"] = sum(d.get("rainMm", 0) or 0 for d in forecast[:7])

        top_crops = get_crop_recommendations(soil_props, weather, top_n=3)

        return jsonify({
            "success": True,
            "recommendations": {
                "summary": f"Based on your {soil_analysis.get('soilType','soil')} soil analysis and current weather in {location}, here are the best crops for maximum profit.",
                "topCrops":            top_crops,
                "intercropSuggestion": f"{top_crops[0]['name']} + legume intercrop for natural nitrogen fixation",
                "keyAlert":            f"{top_crops[0]['name']} shows {top_crops[0]['suitabilityScore']}% suitability — ideal time to prepare fields",
                "nextSteps": [
                    "Collect soil sample for lab test to confirm NPK values",
                    f"Arrange seeds for {top_crops[0]['name']} — best planting window coming",
                    "Apply for PM-KISAN and PMFBY crop insurance before sowing",
                ],
            }
        })

    except Exception as e:
        print(f"[recommendations error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/weather")
def weather():
    try:
        lat      = request.args.get("lat")
        lon      = request.args.get("lon")
        location = request.args.get("location", "your area")

        if not lat or not lon:
            return jsonify({"error": "lat and lon are required"}), 400

        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":      lat,
            "longitude":     lon,
            "current":       "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily":         "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
            "timezone":      "auto",
            "forecast_days": 7,
        }
        resp = requests.get(url, params=params, timeout=10)
        raw  = resp.json()

        current = raw["current"]
        daily   = raw["daily"]

        forecast = []
        for i, date in enumerate(daily["time"]):
            day_label = "Today" if i == 0 else "Tomorrow" if i == 1 else \
                        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][
                            __import__("datetime").datetime.strptime(date, "%Y-%m-%d").weekday()]
            forecast.append({
                "date":       date,
                "day":        day_label,
                "tempMax":    daily["temperature_2m_max"][i],
                "tempMin":    daily["temperature_2m_min"][i],
                "rainChance": daily["precipitation_probability_max"][i],
                "rainMm":     daily["precipitation_sum"][i],
                **wmo_label(daily["weather_code"][i]),
            })

        avg_rain = sum(d["rainChance"] or 0 for d in forecast[:3]) / 3
        advisory = (
            "Heavy rain expected — avoid sowing, check drainage and protect stored crops."
            if avg_rain > 50 else
            "Moderate rainfall likely — good conditions for sowing, monitor for fungal disease."
            if avg_rain > 25 else
            "Dry conditions ahead — ensure irrigation is ready, ideal for land preparation."
        )
        month      = __import__("datetime").datetime.now().month
        season_tag = ("Kharif Season"         if 5 <= month <= 9  else
                      "Rabi Season"           if month >= 10 or month <= 1 else
                      "Pre-Kharif / Zaid Season")

        return jsonify({
            "success": True,
            "weather": {
                "current": {
                    "temp":      round(current["temperature_2m"]),
                    "humidity":  current["relative_humidity_2m"],
                    "windSpeed": current["wind_speed_10m"],
                    **wmo_label(current["weather_code"]),
                },
                "forecast":   forecast,
                "advisory":   advisory,
                "seasonalTag": season_tag,
            }
        })

    except Exception as e:
        print(f"[weather error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/geocode")
def geocode():
    try:
        location = request.args.get("location", "")
        if not location:
            return jsonify({"error": "location is required"}), 400

        url    = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": location, "count": 1, "language": "en", "format": "json"}
        resp   = requests.get(url, params=params, timeout=10)
        data   = resp.json()

        if not data.get("results"):
            return jsonify({"error": "Location not found. Try just the city name."}), 404

        r = data["results"][0]
        return jsonify({
            "success":   True,
            "latitude":  r["latitude"],
            "longitude": r["longitude"],
            "name":      r["name"],
            "country":   r.get("country", ""),
        })

    except Exception as e:
        print(f"[geocode error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/markets")
def markets():
    try:
        location = request.args.get("location", "India")

        distance_premiums = [
            {"name": f"Local Mandi (15 km)",     "premium": 0.00},
            {"name": f"District Market (45 km)", "premium": 0.07},
            {"name": f"City Market (80 km)",     "premium": 0.14},
        ]

        nearby_markets = []
        for m in distance_premiums:
            crops = []
            for crop_name, info in MARKET_PRICES.items():
                adjusted = round(info["price"] * (1 + m["premium"]) * (0.97 + np.random.uniform(0, 0.06)))
                crops.append({
                    "name":     crop_name.capitalize(),
                    "price":    adjusted,
                    "unit":     "quintal",
                    "currency": "INR",
                    "trend":    info["trend"],
                    "demand":   "High" if info["trend"] == "up" else "Medium" if info["trend"] == "stable" else "Low",
                })
            nearby_markets.append({"name": m["name"], "crops": crops})

        return jsonify({
            "success": True,
            "markets": {
                "location":      location,
                "lastUpdated":   __import__("datetime").datetime.utcnow().isoformat(),
                "nearbyMarkets": nearby_markets,
                "dataSource":    "Static prices — integrate Agmarknet API for live data",
            }
        })

    except Exception as e:
        print(f"[markets error] {e}")
        return jsonify({"error": str(e)}), 500


# ─── Start ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"🌾 FarmWise Advisor starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
