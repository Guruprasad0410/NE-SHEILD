import os
import joblib
import pandas as pd
import folium


# =========================================
# PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "mizoram_ml_final_environmental_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "landslide_rf_model.pkl")

OUTPUT_CSV = os.path.join(BASE_DIR, "data", "mizoram_risk_predictions.csv")
OUTPUT_MAP = os.path.join(BASE_DIR, "templates", "risk_map.html")


# =========================================
# LOAD DATA
# =========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))


# =========================================
# LOAD TRAINED MODEL
# =========================================

print("\nLoading trained ML model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully")


# =========================================
# MODEL FEATURES
# =========================================

features = [
    "avg_annual_rainfall_mm",
    "elevation_m",
    "slope_degrees",
    "clay_g_per_kg",
    "sand_g_per_kg",
    "silt_g_per_kg"
]

X = df[features]


# =========================================
# PREDICT SUSCEPTIBILITY SCORE
# =========================================

print("\nGenerating susceptibility scores...")

df["risk_score"] = model.predict_proba(X)[:, 1]

df["risk_percent"] = df["risk_score"] * 100


# =========================================
# RISK CLASSIFICATION
# =========================================

def classify_risk(score):

    if score < 0.25:
        return "Low"

    elif score < 0.50:
        return "Moderate"

    elif score < 0.75:
        return "High"

    else:
        return "Very High"


df["risk_level"] = df["risk_score"].apply(classify_risk)


print("\nRisk distribution:")
print(df["risk_level"].value_counts())


# =========================================
# SAVE PREDICTION CSV
# =========================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nPrediction CSV created:")
print(OUTPUT_CSV)


# =========================================
# CREATE TEMPLATE FOLDER
# =========================================

os.makedirs(
    os.path.join(BASE_DIR, "templates"),
    exist_ok=True
)


# =========================================
# CREATE MIZORAM MAP
# =========================================

print("\nCreating GIS map...")

mizoram_map = folium.Map(

    location=[23.3, 92.8],

    zoom_start=8,

    tiles="OpenStreetMap",

    control_scale=True
)


# =========================================
# CREATE RISK LAYERS
# =========================================

low_layer = folium.FeatureGroup(
    name="🟢 Low Risk",
    show=True
)

moderate_layer = folium.FeatureGroup(
    name="🟡 Moderate Risk",
    show=True
)

high_layer = folium.FeatureGroup(
    name="🟠 High Risk",
    show=True
)

very_high_layer = folium.FeatureGroup(
    name="🔴 Very High Risk",
    show=True
)


low_layer.add_to(mizoram_map)
moderate_layer.add_to(mizoram_map)
high_layer.add_to(mizoram_map)
very_high_layer.add_to(mizoram_map)


# =========================================
# RISK COLORS
# =========================================

colors = {

    "Low": "green",

    "Moderate": "yellow",

    "High": "orange",

    "Very High": "red"
}


# =========================================
# ADD POINTS
# =========================================

print("Adding locations to map...")


for _, row in df.iterrows():

    lat = row["Latitude"]
    lon = row["Longitude"]

    risk = row["risk_level"]

    color = colors[risk]


    popup_html = f"""

    <div style="width:260px">

    <h3 style="
        margin-bottom:5px;
    ">
    NE-SHIELD
    </h3>

    <b>Landslide Susceptibility</b>

    <hr>

    <b>Risk Level:</b>
    {risk}

    <br><br>

    <b>Susceptibility Score:</b>
    {row['risk_percent']:.1f}%

    <hr>

    <b>Rainfall:</b>
    {row['avg_annual_rainfall_mm']:.1f} mm/year

    <br>

    <b>Elevation:</b>
    {row['elevation_m']:.1f} m

    <br>

    <b>Slope:</b>
    {row['slope_degrees']:.1f}°

    <br><br>

    <b>Clay:</b>
    {row['clay_percent']:.1f}%

    <br>

    <b>Sand:</b>
    {row['sand_percent']:.1f}%

    <br>

    <b>Silt:</b>
    {row['silt_percent']:.1f}%

    <hr>

    Prototype susceptibility assessment

    </div>

    """


    marker = folium.CircleMarker(

        location=[
            lat,
            lon
        ],

        radius=4,

        color=color,

        weight=1,

        fill=True,

        fill_color=color,

        fill_opacity=0.65,

        popup=folium.Popup(
            popup_html,
            max_width=300
        ),

        tooltip=f"{risk} Risk | {row['risk_percent']:.1f}%"
    )


    if risk == "Low":

        marker.add_to(low_layer)


    elif risk == "Moderate":

        marker.add_to(moderate_layer)


    elif risk == "High":

        marker.add_to(high_layer)


    else:

        marker.add_to(very_high_layer)


# =========================================
# LEGEND
# =========================================

legend = """

<style>
html, body { margin:0; min-height:100%; background:#f8fafc url('/static/ne-shield-light-bg.png') center/cover fixed no-repeat; font-family:Arial, sans-serif; }
.folium-map { position:fixed !important; top:118px !important; right:32px !important; bottom:32px !important; left:32px !important; width:auto !important; height:auto !important; border:1px solid #cbd5e1; border-radius:12px; box-shadow:0 12px 32px rgba(15,23,42,.14); }
.ne-map-header { position:fixed; top:0; left:0; right:0; height:78px; z-index:9999; display:flex; align-items:center; justify-content:space-between; padding:0 32px; box-sizing:border-box; background:#0b132b; color:#fff; box-shadow:0 2px 10px rgba(15,23,42,.18); }
.ne-map-brand { font-size:20px; font-weight:700; letter-spacing:-.02em; }.ne-map-brand span { color:#93ccff; }
.ne-map-nav { display:flex; gap:5px; align-items:center; }.ne-map-nav a { color:#dbeafe; text-decoration:none; padding:8px 11px; border-radius:6px; font-size:13px; }.ne-map-nav a:hover, .ne-map-nav a.active { background:#007bb9; color:#fff; }
.ne-map-full { border:1px solid #bfdbfe; border-radius:6px; background:#fff; color:#0b1c30; padding:8px 12px; font-weight:600; font-size:13px; cursor:pointer; }
@media (max-width:700px) { .ne-map-header { height:auto; min-height:88px; padding:12px; flex-wrap:wrap; gap:8px; }.ne-map-nav { order:3; width:100%; overflow:auto; }.ne-map-full { display:none; } .folium-map { top:110px !important; right:12px !important; bottom:12px !important; left:12px !important; } }
</style>
<header class="ne-map-header"><a class="ne-map-brand" href="/" style="color:inherit;text-decoration:none">NE-<span>SHIELD</span></a><nav class="ne-map-nav"><a href="/">Dashboard</a><a href="/?view=analyze">Analyze</a><a class="active" href="/map">Risk Map</a><a href="/?view=method">About</a></nav><button class="ne-map-full" onclick="document.documentElement.requestFullscreen && document.documentElement.requestFullscreen()">View full map</button></header>

<div style="
position: fixed;
bottom: 55px;
left: 55px;
width: 220px;
background: white;
border: 2px solid #555;
z-index: 9999;
padding: 15px;
border-radius: 10px;
font-size: 14px;
box-shadow: 0 0 10px rgba(0,0,0,0.3);
">

<b style="font-size:16px;">
NE-SHIELD Risk Level
</b>

<br><br>

<span style="
color:green;
font-size:20px;
">
●
</span>

Low Risk
(&lt;25%)

<br>

<span style="
color:#e6d300;
font-size:20px;
">
●
</span>

Moderate Risk
(25–50%)

<br>

<span style="
color:orange;
font-size:20px;
">
●
</span>

High Risk
(50–75%)

<br>

<span style="
color:red;
font-size:20px;
">
●
</span>

Very High Risk
(≥75%)

<br><br>

<small>
Landslide susceptibility assessment
</small>

</div>

"""


mizoram_map.get_root().html.add_child(
    folium.Element(legend)
)


# =========================================
# LAYER CONTROL
# =========================================

folium.LayerControl(
    collapsed=False
).add_to(mizoram_map)


# =========================================
# SAVE MAP
# =========================================

mizoram_map.save(
    OUTPUT_MAP
)


print("\n==============================")
print("STEP 7A COMPLETED")
print("==============================")

print("\nGIS map saved at:")
print(OUTPUT_MAP)

print("\nRisk CSV saved at:")
print(OUTPUT_CSV)
