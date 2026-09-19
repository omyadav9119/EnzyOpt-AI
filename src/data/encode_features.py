import pandas as pd
from pathlib import Path

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

file = PROJECT_ROOT / "data" / "raw" / "BLAT_ECOLX_Jacquier_2013.csv"

df = pd.read_csv(file)

# -----------------------------------------
# 2. Extract mutation information
# -----------------------------------------

df["original_aa"] = df["mutant"].str[0]
df["position"] = df["mutant"].str[1:-1].astype(int)
df["new_aa"] = df["mutant"].str[-1]

# -----------------------------------------
# 3. One-hot encode amino acids
# -----------------------------------------

original_encoded = pd.get_dummies(
    df["original_aa"],
    prefix="original"
)

new_encoded = pd.get_dummies(
    df["new_aa"],
    prefix="new"
)

# Convert True/False to 0/1
original_encoded = original_encoded.astype(int)
new_encoded = new_encoded.astype(int)

# -----------------------------------------
# 4. Combine features
# -----------------------------------------

features = pd.concat(
    [
        df[["position"]],
        original_encoded,
        new_encoded
    ],
    axis=1
)

# -----------------------------------------
# 5. Add target
# -----------------------------------------

features["DMS_score"] = df["DMS_score"]

# -----------------------------------------
# 6. Display results
# -----------------------------------------

print("=" * 60)
print("ENCODED MUTATION FEATURES")
print("=" * 60)

print("\nFeature shape:")
print(features.shape)

print("\nFeature columns:")
print(features.columns.tolist())

print("\nFirst 10 encoded mutations:")

display_columns = [
    "position",
    "DMS_score"
]

print(
    features[display_columns].head(10).to_string(index=False)
)

print("\nFull feature matrix:")
print(features.head(5).to_string(index=False))

# -----------------------------------------
# 7. Save
# -----------------------------------------

output = PROJECT_ROOT / "data" / "processed" / "encoded_features.csv"

features.to_csv(output, index=False)

print("\nSaved to:")
print(output)