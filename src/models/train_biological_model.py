from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "biological_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — BIOLOGICAL FEATURE MODEL")
print("=" * 60)

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

feature_columns = [
    "position",

    "original_hydrophobicity",
    "new_hydrophobicity",
    "delta_hydrophobicity",

    "original_molecular_weight",
    "new_molecular_weight",
    "delta_molecular_weight",

    "original_polarity",
    "new_polarity",
    "delta_polarity",

    "original_charge",
    "new_charge",
    "delta_charge",
]

X = df[feature_columns]

y = df["DMS_score"]


print("\nFeatures used:")
for feature in feature_columns:
    print(" -", feature)

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining biological model...")

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 7. PREDICT
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 8. EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# 9. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BIOLOGICAL MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# 10. COMPARE WITH BASELINE
# ============================================================

print("\n" + "=" * 60)
print("BASELINE vs BIOLOGICAL MODEL")
print("=" * 60)

print("\nBaseline:")
print("MAE  : 1.3088")
print("RMSE : 1.7010")
print("R²   : 0.3126")

print("\nBiological model:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 11. SHOW PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})

print("\nFirst 20 predictions:")

print(
    results.head(20).to_string(index=False)
)