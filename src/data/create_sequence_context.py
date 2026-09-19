from pathlib import Path

import pandas as pd


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
    / "sequence_context_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — SEQUENCE CONTEXT FEATURES")
print("=" * 60)

print("\nLoading:")
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
# 4. EXTRACT LOCAL SEQUENCE CONTEXT
# ============================================================

def get_context(sequence, position, window=2):
    """
    Extract residues surrounding the mutation.

    The dataset's mutation positions are treated as
    1-based biological positions.

    Returns:
        left_2, left_1, right_1, right_2
    """

    index = position - 1

    left_2 = sequence[index - 2] if index - 2 >= 0 else "X"
    left_1 = sequence[index - 1] if index - 1 >= 0 else "X"

    right_1 = (
        sequence[index + 1]
        if index + 1 < len(sequence)
        else "X"
    )

    right_2 = (
        sequence[index + 2]
        if index + 2 < len(sequence)
        else "X"
    )

    return left_2, left_1, right_1, right_2


contexts = df.apply(
    lambda row: get_context(
        row["mutated_sequence"],
        row["position"]
    ),
    axis=1
)

df[
    [
        "left_2",
        "left_1",
        "right_1",
        "right_2"
    ]
] = pd.DataFrame(
    contexts.tolist(),
    index=df.index
)


# ============================================================
# 5. CREATE FINAL DATASET
# ============================================================

features = df[
    [
        "mutant",
        "position",
        "original_aa",
        "left_2",
        "left_1",
        "right_1",
        "right_2",
        "new_aa",
        "DMS_score"
    ]
].copy()


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\nSequence-context feature shape:")
print(features.shape)

print("\nColumns:")
print(features.columns.tolist())

print("\nFirst 20 mutations with context:")

print(
    features.head(20).to_string(index=False)
)


# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(features.isnull().sum())


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

print("\nSaved sequence context features to:")
print(OUTPUT_FILE)