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

<div style="
position: fixed;
bottom: 40px;
left: 40px;
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
AI-based landslide susceptibility prototype
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