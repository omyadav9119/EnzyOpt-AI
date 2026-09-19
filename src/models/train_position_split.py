from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
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
print("ENZYOPT-AI — POSITION-AWARE MODEL")
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

# Position is used as the grouping variable.
groups = df["position"]

print("\nFeatures:", X.shape[1])
print("Samples:", X.shape[0])
print("Unique positions:", groups.nunique())


# ============================================================
# POSITION-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_positions = groups.iloc[train_idx]
test_positions = groups.iloc[test_idx]


# ============================================================
# VERIFY NO POSITION OVERLAP
# ============================================================

overlap = set(train_positions) & set(test_positions)

print("\nPosition split verification:")
print("Training positions:", train_positions.nunique())
print("Testing positions :", test_positions.nunique())
print("Overlapping positions:", len(overlap))

if len(overlap) != 0:
    raise ValueError(
        "ERROR: Training and testing positions overlap!"
    )

print("No position overlap confirmed.")


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

print("\nTraining position-aware model...")

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


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("POSITION-AWARE MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "position": groups.iloc[test_idx].values,
    "actual": y_test.values,
    "predicted": predictions
})

print("\nFirst 20 predictions:")
print(
    results.head(20).to_string(index=False)
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "position_split_predictions.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPredictions saved to:")
print(OUTPUT_FILE)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print("\nRandom-split final model:")
print("MAE  : 1.0871")
print("RMSE : 1.4493")
print("R²   : 0.5010")

print("\nPosition-aware model:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")