from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENCODED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "encoded_features.csv"
)

BIOLOGICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "biological_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — COMBINED FEATURE GENERATION")
print("=" * 60)

print("\nLoading encoded features:")
print(ENCODED_FILE)

encoded = pd.read_csv(ENCODED_FILE)

print("\nEncoded dataset shape:")
print(encoded.shape)


print("\nLoading biological features:")
print(BIOLOGICAL_FILE)

biological = pd.read_csv(BIOLOGICAL_FILE)

print("\nBiological dataset shape:")
print(biological.shape)


# ============================================================
# 3. CHECK DATA ALIGNMENT
# ============================================================

print("\nChecking mutation alignment...")

if not encoded["DMS_score"].equals(biological["DMS_score"]):
    raise ValueError(
        "DMS_score values do not match between datasets."
    )

print("DMS scores aligned!")

print("\nChecking number of rows...")

if len(encoded) != len(biological):
    raise ValueError(
        "Encoded and biological datasets have different row counts."
    )

print("Row counts match!")


# ============================================================
# 4. SELECT BIOLOGICAL FEATURES
# ============================================================

biological_columns = [
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

biological_features = biological[biological_columns].copy()


# ============================================================
# 5. REMOVE DUPLICATE POSITION
# ============================================================

# encoded_features already contains position.
# Therefore we only add biological properties here.

biological_features = biological_features.drop(
    columns=["position"]
)


# ============================================================
# 6. REMOVE TARGET FROM ENCODED DATA
# ============================================================

encoded_features = encoded.drop(
    columns=["DMS_score"]
)


# ============================================================
# 7. COMBINE FEATURES
# ============================================================

combined = pd.concat(
    [
        encoded_features,
        biological_features
    ],
    axis=1
)


# Add target

combined["DMS_score"] = biological["DMS_score"]


# ============================================================
# 8. CHECK FOR MISSING VALUES
# ============================================================

print("\nChecking missing values...")

missing = combined.isnull().sum().sum()

print("Total missing values:", missing)

if missing > 0:
    raise ValueError(
        "Combined dataset contains missing values."
    )


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\nCombined dataset shape:")
print(combined.shape)

print("\nNumber of features:")
print(combined.shape[1] - 1)

print("\nFeature columns:")

for column in combined.columns:
    print(" -", column)


# ============================================================
# 10. SAVE
# ============================================================

combined.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nSaved combined dataset to:")
print(OUTPUT_FILE)