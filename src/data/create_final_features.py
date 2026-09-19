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

SEQUENCE_CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "encoded_sequence_context.csv"
)

CONTEXT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "context_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — FINAL FEATURE GENERATION")
print("=" * 60)


print("\nLoading mutation features:")
print(ENCODED_FILE)

encoded = pd.read_csv(ENCODED_FILE)

print("Shape:", encoded.shape)


print("\nLoading biological features:")
print(BIOLOGICAL_FILE)

biological = pd.read_csv(BIOLOGICAL_FILE)

print("Shape:", biological.shape)


print("\nLoading sequence-context features:")
print(SEQUENCE_CONTEXT_FILE)

sequence_context = pd.read_csv(SEQUENCE_CONTEXT_FILE)

print("Shape:", sequence_context.shape)


print("\nLoading local context features:")
print(CONTEXT_FILE)

context = pd.read_csv(CONTEXT_FILE)

print("Shape:", context.shape)


# ============================================================
# 3. CHECK ROW ALIGNMENT
# ============================================================

print("\nChecking row alignment...")


# Row counts
if not (
    len(encoded)
    == len(biological)
    == len(sequence_context)
    == len(context)
):
    raise ValueError(
        "Dataset row counts do not match!"
    )

print("Row counts match!")


# DMS score alignment
if not (
    encoded["DMS_score"].equals(
        biological["DMS_score"]
    )
    and encoded["DMS_score"].equals(
        sequence_context["DMS_score"]
    )
    and encoded["DMS_score"].equals(
        context["DMS_score"]
    )
):
    raise ValueError(
        "DMS scores are not aligned!"
    )

print("All DMS scores aligned!")


# Position alignment
if not (
    encoded["position"].equals(
        biological["position"]
    )
    and encoded["position"].equals(
        sequence_context["position"]
    )
    and encoded["position"].equals(
        context["position"]
    )
):
    raise ValueError(
        "Mutation positions are not aligned!"
    )

print("Mutation positions aligned!")


# ============================================================
# 4. SELECT MUTATION FEATURES
# ============================================================

print("\nSelecting mutation features...")

mutation_features = encoded.drop(
    columns=[
        "DMS_score"
    ],
    errors="ignore"
)

print(
    "Mutation feature shape:",
    mutation_features.shape
)


# ============================================================
# 5. SELECT BIOLOGICAL FEATURES
# ============================================================

print("\nSelecting biological features...")

biological_features = biological.drop(
    columns=[
        "mutant",
        "original_aa",
        "new_aa",
        "DMS_score",
        "position"
    ],
    errors="ignore"
)

print(
    "Biological feature shape:",
    biological_features.shape
)


# ============================================================
# 6. SELECT SEQUENCE-CONTEXT FEATURES
# ============================================================

print("\nSelecting sequence-context features...")

# encoded_sequence_context.csv contains:
#
# position
# original_aa
# new_aa
# DMS_score
# left_2_*
# left_1_*
# right_1_*
# right_2_*
#
# It does NOT contain "mutant".
#
# Therefore only remove columns that are actually present.

sequence_context_features = sequence_context.drop(
    columns=[
        "position",
        "original_aa",
        "new_aa",
        "DMS_score"
    ],
    errors="ignore"
)

print(
    "Sequence-context feature shape:",
    sequence_context_features.shape
)


# ============================================================
# 7. SELECT LOCAL CONTEXT FEATURES
# ============================================================

print("\nSelecting local context features...")

# context_features.csv contains:
#
# mutant
# position
# original_aa
# new_aa
# local_*
# new_vs_local_*
# DMS_score
#
# Keep only the actual numerical context features.

context_features = context.drop(
    columns=[
        "mutant",
        "position",
        "original_aa",
        "new_aa",
        "DMS_score"
    ],
    errors="ignore"
)

print(
    "Local context feature shape:",
    context_features.shape
)


# ============================================================
# 8. COMBINE FEATURES
# ============================================================

print("\nCombining feature sets...")

final_features = pd.concat(
    [
        mutation_features,
        biological_features,
        sequence_context_features,
        context_features
    ],
    axis=1
)


# ============================================================
# 9. ADD TARGET
# ============================================================

final_features["DMS_score"] = (
    encoded["DMS_score"]
)


# ============================================================
# 10. REMOVE DUPLICATED COLUMNS
# ============================================================

duplicated_columns = (
    final_features.columns[
        final_features.columns.duplicated()
    ]
)

if len(duplicated_columns) > 0:

    print(
        "\nRemoving duplicated columns:"
    )

    for column in duplicated_columns:
        print(" -", column)

    final_features = final_features.loc[
        :,
        ~final_features.columns.duplicated()
    ]


# ============================================================
# 11. CHECK DATA
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET CHECK")
print("=" * 60)


print("\nFinal dataset shape:")
print(final_features.shape)


print("\nNumber of features:")
print(
    final_features.shape[1] - 1
)


print("\nTarget column:")
print(
    "DMS_score"
)


missing_values = (
    final_features.isnull().sum().sum()
)

print("\nTotal missing values:")
print(missing_values)


duplicate_count = (
    final_features.columns.duplicated().sum()
)

print("\nNumber of duplicated columns:")
print(duplicate_count)


# ============================================================
# 12. VALIDATION CHECKS
# ============================================================

if missing_values != 0:
    raise ValueError(
        "Final dataset contains missing values!"
    )

if duplicate_count != 0:
    raise ValueError(
        "Final dataset contains duplicated columns!"
    )

if "DMS_score" not in final_features.columns:
    raise ValueError(
        "DMS_score target column is missing!"
    )


# ============================================================
# 13. SAVE
# ============================================================

final_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 14. SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nSaved final feature dataset to:")
print(OUTPUT_FILE)