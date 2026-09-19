from pathlib import Path
import pandas as pd

from amino_acid_properties import AMINO_ACID_PROPERTIES


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "BLAT_ECOLX_Jacquier_2013.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "biological_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — BIOLOGICAL FEATURE GENERATION")
print("=" * 60)

print("\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. EXTRACT MUTATION INFORMATION
# ============================================================

df["original_aa"] = df["mutant"].str[0]
df["position"] = df["mutant"].str[1:-1].astype(int)
df["new_aa"] = df["mutant"].str[-1]


# ============================================================
# 4. GET BIOLOGICAL PROPERTIES
# ============================================================

df["original_hydrophobicity"] = df["original_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["hydrophobicity"]
)

df["new_hydrophobicity"] = df["new_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["hydrophobicity"]
)

df["original_molecular_weight"] = df["original_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["molecular_weight"]
)

df["new_molecular_weight"] = df["new_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["molecular_weight"]
)

df["original_polarity"] = df["original_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["polarity"]
)

df["new_polarity"] = df["new_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["polarity"]
)

df["original_charge"] = df["original_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["charge"]
)

df["new_charge"] = df["new_aa"].map(
    lambda aa: AMINO_ACID_PROPERTIES[aa]["charge"]
)


# ============================================================
# 5. CALCULATE CHANGES (DELTA FEATURES)
# ============================================================

df["delta_hydrophobicity"] = (
    df["new_hydrophobicity"]
    - df["original_hydrophobicity"]
)

df["delta_molecular_weight"] = (
    df["new_molecular_weight"]
    - df["original_molecular_weight"]
)

df["delta_polarity"] = (
    df["new_polarity"]
    - df["original_polarity"]
)

df["delta_charge"] = (
    df["new_charge"]
    - df["original_charge"]
)


# ============================================================
# 6. SELECT FINAL FEATURES
# ============================================================

features = df[
    [
        "mutant",
        "position",
        "original_aa",
        "new_aa",

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

        "DMS_score"
    ]
].copy()


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\nBiological feature shape:")
print(features.shape)

print("\nColumns:")
print(features.columns.tolist())

print("\nFirst 10 mutations:")
print(
    features.head(10).to_string(index=False)
)


# ============================================================
# 8. SAVE
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nSaved biological features to:")
print(OUTPUT_FILE)