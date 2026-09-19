from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — FINAL MODEL")
print("=" * 60)

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(columns=["DMS_score"])
y = df["DMS_score"]

print("\nFeatures:", X.shape[1])
print("Samples:", X.shape[0])


# ============================================================
# TRAIN / TEST SPLIT
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
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=None
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining final model...")

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# PERFORMANCE
# ============================================================

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": predictions
})

print("\nFirst 20 predictions:")
print(results.head(20).to_string(index=False))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 20 important features:")
print(
    importance.head(20).to_string(index=False)
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_importance.csv"
)

importance.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFeature importance saved to:")
print(OUTPUT_FILE)