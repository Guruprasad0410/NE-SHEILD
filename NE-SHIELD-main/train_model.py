import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# LOAD DATA
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "mizoram_ml_final_environmental_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))

# -----------------------------
# FEATURES
# -----------------------------

features = [
    "avg_annual_rainfall_mm",
    "elevation_m",
    "slope_degrees",
    "clay_g_per_kg",
    "sand_g_per_kg",
    "silt_g_per_kg"
]

X = df[features]
y = df["label"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=2,
    n_jobs=-1
)

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed")

# -----------------------------
# TEST MODEL
# -----------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print("\nMODEL RESULTS")
print("-----------------------------")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df.to_string(index=False))

# -----------------------------
# SAVE TRAINED MODEL
# -----------------------------

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "models", "landslide_rf_model.pkl")

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully:")
print(MODEL_PATH)