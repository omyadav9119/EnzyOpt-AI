from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# ENZYOPT-AI — FEATURE ABLATION
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — FEATURE ABLATION")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_ablation_results.csv"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_ablation_predictions.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("\nSamples:", len(df))
print("Features:", df.shape[1] - 1)
print("Unique positions:", df["position"].nunique())


# ============================================================
# 4. TARGET / GROUPS
# ============================================================

y = df["DMS_score"]

# Position is used ONLY for grouping.
#
# It is intentionally NOT used as a predictive feature.
# This allows the ablation experiment to measure how well
# different feature groups generalize to unseen positions.

groups = df["position"]


# ============================================================
# 5. DEFINE FEATURE GROUPS
# ============================================================

# ------------------------------------------------------------
# Mutation features
# ------------------------------------------------------------
#
# Mutation features are the one-hot encoded mutation identity
# features:
#
#     original_A, original_C, ...
#     new_A, new_C, ...
#
# Biological numerical properties are explicitly excluded.
#
# Local-context comparison features are also explicitly
# excluded so they remain ONLY in the local-context group.
# ------------------------------------------------------------

biological_property_columns = {
    "original_hydrophobicity",
    "original_polarity",
    "original_charge",
    "original_molecular_weight",
    "new_hydrophobicity",
    "new_polarity",
    "new_charge",
    "new_molecular_weight",
}

local_comparison_columns = {
    "new_vs_local_hydrophobicity",
    "new_vs_local_molecular_weight",
    "new_vs_local_polarity",
    "new_vs_local_charge",
}

mutation_columns = [
    column
    for column in df.columns
    if (
        (
            column.startswith("original_")
            or column.startswith("new_")
        )
        and column not in biological_property_columns
        and column not in local_comparison_columns
    )
]


# ------------------------------------------------------------
# Biological features
# ------------------------------------------------------------
#
# These describe intrinsic amino-acid properties and the
# physicochemical change caused by the mutation.
# ------------------------------------------------------------

biological_columns = [
    "original_hydrophobicity",
    "original_polarity",
    "original_charge",
    "original_molecular_weight",

    "new_hydrophobicity",
    "new_polarity",
    "new_charge",
    "new_molecular_weight",

    "delta_hydrophobicity",
    "delta_polarity",
    "delta_charge",
    "delta_molecular_weight",
]


# ------------------------------------------------------------
# Sequence-context features
# ------------------------------------------------------------
#
# These are the one-hot encoded residues surrounding the
# mutation site.
# ------------------------------------------------------------

sequence_context_columns = [
    column
    for column in df.columns
    if (
        column.startswith("left_")
        or column.startswith("right_")
    )
]


# ------------------------------------------------------------
# Local-context features
# ------------------------------------------------------------
#
# These describe the physicochemical environment surrounding
# the mutation and compare the new amino acid against that
# local environment.
# ------------------------------------------------------------

local_context_columns = [
    column
    for column in df.columns
    if (
        column.startswith("local_")
        or column.startswith("new_vs_local_")
    )
]


# ============================================================
# 6. VERIFY FEATURE GROUPS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE GROUPS")
print("=" * 60)

print(
    "\nMutation features:",
    len(mutation_columns),
)

print(
    "Biological features:",
    len(biological_columns),
)

print(
    "Sequence-context features:",
    len(sequence_context_columns),
)

print(
    "Local-context features:",
    len(local_context_columns),
)


# ------------------------------------------------------------
# Check required biological features
# ------------------------------------------------------------

missing_biological = [
    column
    for column in biological_columns
    if column not in df.columns
]

if missing_biological:
    raise ValueError(
        "Missing biological features: "
        + str(missing_biological)
    )


# ------------------------------------------------------------
# Check required local-context features
# ------------------------------------------------------------

missing_local = [
    column
    for column in local_context_columns
    if column not in df.columns
]

if missing_local:
    raise ValueError(
        "Missing local-context features: "
        + str(missing_local)
    )


# ------------------------------------------------------------
# Check feature-group overlap
# ------------------------------------------------------------

all_group_columns = (
    mutation_columns
    + biological_columns
    + sequence_context_columns
    + local_context_columns
)

all_group_series = pd.Series(
    all_group_columns
)

duplicates = all_group_series[
    all_group_series.duplicated()
].tolist()

if duplicates:
    raise ValueError(
        "Feature groups overlap: "
        + str(duplicates)
    )

print("\nNo feature-group overlap confirmed.")


# ------------------------------------------------------------
# Check that every predictive feature is assigned exactly once
# ------------------------------------------------------------

excluded_columns = {
    "position",
    "DMS_score",
}

expected_features = (
    set(df.columns) - excluded_columns
)

grouped_features = set(
    all_group_columns
)


missing_from_groups = sorted(
    expected_features - grouped_features
)

unexpected_features = sorted(
    grouped_features - expected_features
)


if missing_from_groups:

    print(
        "\nWARNING — features not assigned to a group:"
    )

    for feature in missing_from_groups:
        print(
            f"  - {feature}"
        )


if unexpected_features:

    raise ValueError(
        "Unexpected features in feature groups: "
        + str(unexpected_features)
    )


if not missing_from_groups:

    print(
        "All predictive features assigned to exactly one group."
    )


# ============================================================
# 7. CREATE FEATURE SETS
# ============================================================

feature_sets = {

    "Mutation only": (
        mutation_columns
    ),

    "Mutation + Biological": (
        mutation_columns
        + biological_columns
    ),

    "Mutation + Sequence": (
        mutation_columns
        + sequence_context_columns
    ),

    "Mutation + Local Context": (
        mutation_columns
        + local_context_columns
    ),

    "All features": (
        mutation_columns
        + biological_columns
        + sequence_context_columns
        + local_context_columns
    ),
}


# ============================================================
# 8. VERIFY FEATURE SETS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SETS")
print("=" * 60)

for name, columns in feature_sets.items():

    print(
        f"\n{name}: {len(columns)} features"
    )

    if len(columns) != len(set(columns)):

        duplicate_features = pd.Series(
            columns
        )[
            pd.Series(columns).duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate features in '{name}': "
            + str(duplicate_features)
        )


# ============================================================
# 9. POSITION-AWARE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("POSITION-AWARE SPLIT")
print("=" * 60)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        y,
        groups=groups,
    )
)


train_positions = set(
    groups.iloc[train_idx]
)

test_positions = set(
    groups.iloc[test_idx]
)

overlap = (
    train_positions
    & test_positions
)


print(
    "\nTraining samples:",
    len(train_idx),
)

print(
    "Testing samples:",
    len(test_idx),
)

print(
    "Training positions:",
    len(train_positions),
)

print(
    "Testing positions:",
    len(test_positions),
)

print(
    "Overlapping positions:",
    len(overlap),
)


if overlap:

    raise ValueError(
        "Position leakage detected!"
    )


print(
    "No position overlap confirmed."
)


# ============================================================
# 10. TRAIN / EVALUATE FUNCTION
# ============================================================

def evaluate_feature_set(
    name,
    columns,
):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    X = df[columns]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(
        "Number of features:",
        X.shape[1],
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "feature_set": name,
        "n_features": X.shape[1],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# 11. RUN ABLATION
# ============================================================

print("\n" + "=" * 60)
print("RUNNING FEATURE ABLATION")
print("=" * 60)

results = []

for name, columns in feature_sets.items():

    result = evaluate_feature_set(
        name,
        columns,
    )

    results.append(result)


# ============================================================
# 12. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("FEATURE ABLATION RESULTS")
print("=" * 60)

print(
    "\n"
    + results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

prediction_rows = []

for name, columns in feature_sets.items():

    X = df[columns]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    for i, prediction in zip(
        test_idx,
        predictions,
    ):

        prediction_rows.append(
            {
                "feature_set": name,
                "position": df.iloc[i]["position"],
                "actual": df.iloc[i]["DMS_score"],
                "predicted": prediction,
            }
        )


prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
)


# ============================================================
# 15. SUCCESS
# ============================================================

print("\n" + "=" * 60)
print("ABLATION COMPLETE")
print("=" * 60)

print(
    "\nResults saved to:"
)

print(OUTPUT_FILE)

print(
    "\nPredictions saved to:"
)

print(PREDICTION_FILE)