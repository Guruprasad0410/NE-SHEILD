from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import math


app = Flask(__name__)


# =========================================
# FILE PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RISK_DATA = os.path.join(BASE_DIR, "data", "mizoram_risk_predictions.csv")

ENV_DATA = os.path.join(BASE_DIR, "data", "mizoram_ml_final_environmental_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "landslide_rf_model.pkl")


# =========================================
# LOAD MODEL
# =========================================

print("Loading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# =========================================
# LOAD ENVIRONMENT DATASET
# =========================================

print("Loading environmental dataset...")

environment_df = pd.read_csv(ENV_DATA)

print("Environmental dataset loaded successfully.")

print(
    "Environmental rows:",
    len(environment_df)
)


# =========================================
# MODEL FEATURES
# =========================================

FEATURES = [

    "avg_annual_rainfall_mm",

    "elevation_m",

    "slope_degrees",

    "clay_g_per_kg",

    "sand_g_per_kg",

    "silt_g_per_kg"
]


# =========================================
# STATIC LANDSLIDE SUSCEPTIBILITY
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


# =========================================
# RECENT RAINFALL TRIGGER
#
# Prototype demonstration thresholds.
# These are NOT official operational
# GSI rainfall thresholds.
# =========================================

def rainfall_trigger(
    rain_24h,
    rain_72h
):

    if (
        rain_24h >= 150
        or
        rain_72h >= 300
    ):

        return "Very High"


    elif (
        rain_24h >= 100
        or
        rain_72h >= 200
    ):

        return "High"


    elif (
        rain_24h >= 50
        or
        rain_72h >= 100
    ):

        return "Moderate"


    else:

        return "Low"


# =========================================
# FINAL EARLY WARNING MATRIX
# =========================================

def final_warning_level(
    susceptibility,
    rainfall_level
):

    warning_matrix = {

        "Low": {

            "Low": "Low",

            "Moderate": "Low",

            "High": "Moderate",

            "Very High": "High"
        },


        "Moderate": {

            "Low": "Low",

            "Moderate": "Moderate",

            "High": "High",

            "Very High": "Very High"
        },


        "High": {

            "Low": "Moderate",

            "Moderate": "High",

            "High": "Very High",

            "Very High": "Very High"
        },


        "Very High": {

            "Low": "High",

            "Moderate": "High",

            "High": "Very High",

            "Very High": "Very High"
        }
    }


    return warning_matrix[
        susceptibility
    ][
        rainfall_level
    ]


# =========================================
# HAVERSINE DISTANCE
# Returns distance in kilometres
# =========================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0


    lat1 = math.radians(
        lat1
    )

    lon1 = math.radians(
        lon1
    )

    lat2 = math.radians(
        lat2
    )

    lon2 = math.radians(
        lon2
    )


    dlat = lat2 - lat1

    dlon = lon2 - lon1


    a = (

        math.sin(
            dlat / 2
        ) ** 2

        +

        math.cos(
            lat1
        )

        *

        math.cos(
            lat2
        )

        *

        math.sin(
            dlon / 2
        ) ** 2
    )


    c = 2 * math.atan2(

        math.sqrt(
            a
        ),

        math.sqrt(
            1 - a
        )
    )


    distance = (
        earth_radius
        *
        c
    )


    return distance


# =========================================
# FIND NEAREST ENVIRONMENTAL POINT
# =========================================

def find_nearest_environment(
    latitude,
    longitude
):

    temp_df = environment_df.copy()


    temp_df[
        "distance_km"
    ] = temp_df.apply(

        lambda row:

            haversine_distance(

                latitude,

                longitude,

                row["Latitude"],

                row["Longitude"]
            ),

        axis=1
    )


    nearest_index = temp_df[
        "distance_km"
    ].idxmin()


    nearest = temp_df.loc[
        nearest_index
    ]


    return nearest


# =========================================
# EMERGENCY PRIORITY ENGINE
#
# Prototype decision support logic.
# Not an official emergency triage system.
# =========================================

def calculate_emergency_priority(

    warning_level,

    road_affected,

    village_nearby,

    critical_facility,

    people_exposed
):

    score = 0


    # =====================================
    # WARNING CONTRIBUTION
    # =====================================

    warning_scores = {

        "Low": 0,

        "Moderate": 2,

        "High": 4,

        "Very High": 6
    }


    score += warning_scores.get(
        warning_level,
        0
    )


    # =====================================
    # ROAD IMPACT
    # =====================================

    if road_affected == "yes":

        score += 2


    # =====================================
    # SETTLEMENT IMPACT
    # =====================================

    if village_nearby == "yes":

        score += 2


    # =====================================
    # CRITICAL FACILITY
    # =====================================

    if critical_facility == "yes":

        score += 3


    # =====================================
    # PEOPLE EXPOSED
    # =====================================

    if people_exposed >= 100:

        score += 3


    elif people_exposed >= 50:

        score += 2


    elif people_exposed > 0:

        score += 1


    # =====================================
    # FINAL PRIORITY
    # =====================================

    if score >= 11:

        priority = "CRITICAL"

        action = (

            "Immediate field verification, evacuation readiness, "

            "traffic restriction and emergency response team deployment."
        )


    elif score >= 8:

        priority = "HIGH"

        action = (

            "Alert local authorities, inspect affected infrastructure "

            "and prepare emergency response teams."
        )


    elif score >= 4:

        priority = "MEDIUM"

        action = (

            "Increase monitoring and conduct field verification."
        )


    else:

        priority = "LOW"

        action = (

            "Continue routine monitoring."
        )


    return (
        priority,
        action,
        score
    )


# =========================================
# MAIN DASHBOARD
# =========================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)

def dashboard():

    # =====================================
    # LOAD MAP RISK COUNTS
    # =====================================

    df = pd.read_csv(
        RISK_DATA
    )


    counts = df[
        "risk_level"
    ].value_counts()


    low = int(
        counts.get(
            "Low",
            0
        )
    )


    moderate = int(
        counts.get(
            "Moderate",
            0
        )
    )


    high = int(
        counts.get(
            "High",
            0
        )
    )


    very_high = int(
        counts.get(
            "Very High",
            0
        )
    )


    total = len(
        df
    )


    prediction = None


    # =====================================
    # ANALYZE LOCATION
    # =====================================

    if request.method == "POST":

        try:

            # =================================
            # LOCATION INPUT
            # =================================

            latitude = float(
                request.form[
                    "latitude"
                ]
            )


            longitude = float(
                request.form[
                    "longitude"
                ]
            )


            # =================================
            # PROTOTYPE BOUNDARY CHECK
            # =================================

            if (
                latitude < 21.0
                or
                latitude > 25.5
            ):

                raise ValueError(

                    "Latitude appears outside the Mizoram prototype area."
                )


            if (
                longitude < 92.0
                or
                longitude > 94.0
            ):

                raise ValueError(

                    "Longitude appears outside the Mizoram prototype area."
                )


            # =================================
            # AUTO ENVIRONMENT LOOKUP
            # =================================

            nearest = (
                find_nearest_environment(
                    latitude,
                    longitude
                )
            )


            rainfall = float(
                nearest[
                    "avg_annual_rainfall_mm"
                ]
            )


            elevation = float(
                nearest[
                    "elevation_m"
                ]
            )


            slope = float(
                nearest[
                    "slope_degrees"
                ]
            )


            clay = float(
                nearest[
                    "clay_g_per_kg"
                ]
            )


            sand = float(
                nearest[
                    "sand_g_per_kg"
                ]
            )


            silt = float(
                nearest[
                    "silt_g_per_kg"
                ]
            )


            nearest_latitude = float(
                nearest[
                    "Latitude"
                ]
            )


            nearest_longitude = float(
                nearest[
                    "Longitude"
                ]
            )


            nearest_distance = float(
                nearest[
                    "distance_km"
                ]
            )


            # =================================
            # RECENT RAINFALL INPUT
            # =================================

            rain_24h = float(

                request.form[
                    "rain_24h"
                ]
            )


            rain_72h = float(

                request.form[
                    "rain_72h"
                ]
            )


            # =================================
            # IMPACT INPUT
            # =================================

            road_affected = (
                request.form.get(
                    "road_affected",
                    "no"
                )
            )


            village_nearby = (
                request.form.get(
                    "village_nearby",
                    "no"
                )
            )


            critical_facility = (
                request.form.get(
                    "critical_facility",
                    "no"
                )
            )


            people_exposed = int(

                request.form.get(
                    "people_exposed",
                    0
                )
            )


            # =================================
            # VALIDATION
            # =================================

            if (
                rain_24h < 0
                or
                rain_72h < 0
            ):

                raise ValueError(

                    "Recent rainfall cannot be negative."
                )


            if people_exposed < 0:

                raise ValueError(

                    "People exposed cannot be negative."
                )


            # =================================
            # RANDOM FOREST MODEL INPUT
            # =================================

            input_data = pd.DataFrame(

                [[

                    rainfall,

                    elevation,

                    slope,

                    clay,

                    sand,

                    silt

                ]],

                columns=FEATURES
            )


            probability = model.predict_proba(
                input_data
            )[0][1]


            susceptibility_score = (

                probability
                *
                100
            )


            susceptibility_level = (
                classify_risk(
                    probability
                )
            )


            # =================================
            # RAINFALL TRIGGER
            # =================================

            rainfall_level = (
                rainfall_trigger(
                    rain_24h,
                    rain_72h
                )
            )


            # =================================
            # FINAL WARNING
            # =================================

            warning_level = (
                final_warning_level(

                    susceptibility_level,

                    rainfall_level
                )
            )


            # =================================
            # EMERGENCY PRIORITY
            # =================================

            (

                emergency_priority,

                recommended_action,

                priority_score

            ) = calculate_emergency_priority(

                warning_level,

                road_affected,

                village_nearby,

                critical_facility,

                people_exposed
            )


            # =================================
            # RESULT
            # =================================

            prediction = {


                # Selected coordinate

                "latitude": round(
                    latitude,
                    6
                ),


                "longitude": round(
                    longitude,
                    6
                ),


                # Nearest environment point

                "nearest_latitude": round(
                    nearest_latitude,
                    6
                ),


                "nearest_longitude": round(
                    nearest_longitude,
                    6
                ),


                "nearest_distance": round(
                    nearest_distance,
                    2
                ),


                # Environmental values

                "rainfall": round(
                    rainfall,
                    2
                ),


                "elevation": round(
                    elevation,
                    2
                ),


                "slope": round(
                    slope,
                    2
                ),


                "clay": round(
                    clay,
                    2
                ),


                "sand": round(
                    sand,
                    2
                ),


                "silt": round(
                    silt,
                    2
                ),


                # ML result

                "score": round(
                    susceptibility_score,
                    2
                ),


                "risk": (
                    susceptibility_level
                ),


                # Rainfall

                "rain_24h": (
                    rain_24h
                ),


                "rain_72h": (
                    rain_72h
                ),


                "rainfall_trigger": (
                    rainfall_level
                ),


                # Early warning

                "warning": (
                    warning_level
                ),


                # Impact

                "road_affected": (
                    road_affected
                ),


                "village_nearby": (
                    village_nearby
                ),


                "critical_facility": (
                    critical_facility
                ),


                "people_exposed": (
                    people_exposed
                ),


                # Emergency response

                "priority": (
                    emergency_priority
                ),


                "priority_score": (
                    priority_score
                ),


                "recommended_action": (
                    recommended_action
                )
            }


            # =================================
            # TERMINAL OUTPUT
            # =================================

            print(
                "\n================================"
            )

            print(
                "NE-SHIELD LOCATION ANALYSIS"
            )

            print(
                "================================"
            )


            print(

                "Requested coordinate:",

                latitude,

                longitude
            )


            print(

                "Nearest environmental point:",

                nearest_latitude,

                nearest_longitude
            )


            print(

                "Nearest distance:",

                round(
                    nearest_distance,
                    2
                ),

                "km"
            )


            print(

                "Annual rainfall:",

                round(
                    rainfall,
                    2
                ),

                "mm/year"
            )


            print(

                "Elevation:",

                round(
                    elevation,
                    2
                ),

                "m"
            )


            print(

                "Slope:",

                round(
                    slope,
                    2
                ),

                "degrees"
            )


            print(

                "ML susceptibility:",

                susceptibility_level
            )


            print(

                "ML score:",

                round(
                    susceptibility_score,
                    2
                ),

                "%"
            )


            print(

                "Rainfall trigger:",

                rainfall_level
            )


            print(

                "FINAL WARNING:",

                warning_level
            )


            print(

                "Emergency Priority:",

                emergency_priority
            )


            print(

                "Priority Score:",

                priority_score
            )


            print(
                "================================\n"
            )


        except Exception as e:

            prediction = {

                "error": str(
                    e
                )
            }


    return render_template(

        "index.html",

        total=total,

        low=low,

        moderate=moderate,

        high=high,

        very_high=very_high,

        prediction=prediction
    )


# =========================================
# GIS MAP
#
# This route loads your existing Folium map.
# If selected coordinates are supplied,
# JavaScript injects another marker.
# =========================================

@app.route(
    "/map"
)

def risk_map():

    latitude = request.args.get(
        "lat"
    )


    longitude = request.args.get(
        "lon"
    )


    risk = request.args.get(
        "risk",
        "N/A"
    )


    warning = request.args.get(
        "warning",
        "N/A"
    )


    priority = request.args.get(
        "priority",
        "N/A"
    )


    map_path = os.path.join(

        app.root_path,

        "templates",

        "risk_map.html"
    )


    with open(

        map_path,

        "r",

        encoding="utf-8"

    ) as file:

        map_html = file.read()


    # =====================================
    # NORMAL MAP WHEN NO LOCATION SELECTED
    # =====================================

    if (
        latitude is None
        or
        longitude is None
    ):

        return map_html


    try:

        latitude = float(
            latitude
        )


        longitude = float(
            longitude
        )


        # Safe text for JavaScript popup

        popup_text = f"""

            <div style='min-width:220px;'>

                <h4 style='margin-bottom:8px;'>
                    Selected Analysis Location
                </h4>

                <b>Latitude:</b>
                {latitude}

                <br>

                <b>Longitude:</b>
                {longitude}

                <br><br>

                <b>Susceptibility:</b>
                {risk}

                <br>

                <b>Early Warning:</b>
                {warning}

                <br>

                <b>Emergency Priority:</b>
                {priority}

            </div>

        """


        marker_script = f"""

        <script>

        window.addEventListener(
            "load",
            function() {{

                setTimeout(
                    function() {{

                        try {{

                            var selectedMap = null;


                            for (
                                var key in window
                            ) {{

                                if (

                                    key.startsWith(
                                        "map_"
                                    )

                                    &&

                                    window[key]

                                    &&

                                    typeof window[key].setView
                                    ===
                                    "function"

                                ) {{

                                    selectedMap = window[key];

                                    break;

                                }}

                            }}


                            if (
                                selectedMap
                            ) {{

                                var selectedIcon = L.divIcon({{

                                    className: "",

                                    html: `

                                        <div style="
                                            width:28px;
                                            height:28px;
                                            background:#2563eb;
                                            border:4px solid white;
                                            border-radius:50%;
                                            box-shadow:0 0 0 3px #1e40af;
                                        ">
                                        </div>

                                    `,

                                    iconSize: [
                                        28,
                                        28
                                    ],

                                    iconAnchor: [
                                        14,
                                        14
                                    ]

                                }});


                                var selectedMarker = L.marker(

                                    [
                                        {latitude},
                                        {longitude}
                                    ],

                                    {{

                                        icon:
                                        selectedIcon

                                    }}

                                ).addTo(
                                    selectedMap
                                );


                                selectedMarker.bindPopup(

                                    `{popup_text}`

                                ).openPopup();


                                selectedMap.setView(

                                    [
                                        {latitude},
                                        {longitude}
                                    ],

                                    11
                                );

                            }}


                        }}

                        catch (
                            error
                        ) {{

                            console.log(

                                "Selected marker error:",

                                error
                            );

                        }}

                    }},

                    500
                );

            }}
        );

        </script>

        """


        map_html = map_html.replace(

            "</body>",

            marker_script
            +
            "</body>"
        )


        return map_html


    except Exception as e:

        print(

            "Map marker error:",

            e
        )


        return map_html


# =========================================
# START SERVER
# =========================================

if __name__ == "__main__":

    print(
        "\n===================================="
    )

    print(
        "             NE-SHIELD"
    )

    print(
        "===================================="
    )

    print(
        "AI-Powered Landslide Risk Intelligence"
    )

    print(
        "Early Warning & Emergency Decision Support"
    )

    print(
        "------------------------------------"
    )


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )