from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequence_context_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "encoded_sequence_context.csv"
)


# ============================================================
# 2. AMINO ACIDS
# ============================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

CONTEXT_COLUMNS = [
    "left_2",
    "left_1",
    "right_1",
    "right_2"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — ENCODE SEQUENCE CONTEXT")
print("=" * 60)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 4. ONE-HOT ENCODE CONTEXT
# ============================================================

encoded_parts = []

for column in CONTEXT_COLUMNS:

    encoded = pd.get_dummies(
        df[column],
        prefix=column
    )

    # Make sure every amino acid has a column
    expected_columns = [
        f"{column}_{aa}"
        for aa in AMINO_ACIDS
    ]

    for expected in expected_columns:
        if expected not in encoded.columns:
            encoded[expected] = 0

    encoded = encoded[expected_columns]

    encoded_parts.append(encoded.astype(int))


# ============================================================
# 5. COMBINE FEATURES
# ============================================================

base_features = df[
    [
        "position",
        "original_aa",
        "new_aa",
        "DMS_score"
    ]
].copy()

context_encoded = pd.concat(
    encoded_parts,
    axis=1
)

result = pd.concat(
    [
        base_features,
        context_encoded
    ],
    axis=1
)


# ============================================================
# 6. DISPLAY
# ============================================================

print("\nEncoded dataset shape:")
print(result.shape)

print("\nColumns:")
print(result.columns.tolist())

print("\nFirst 10 encoded mutations:")

print(
    result.head(10).to_string(index=False)
)


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(result.isnull().sum().sum())


# ============================================================
# 8. SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nSaved encoded sequence context to:")
print(OUTPUT_FILE)