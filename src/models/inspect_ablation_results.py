from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# ENZYOPT-AI — ABLATION ERROR ANALYSIS
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — ABLATION ERROR ANALYSIS")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_ablation_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_ablation_predictions.csv"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)

ERROR_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ablation_error_summary.csv"
)

POSITION_ERROR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ablation_position_error.csv"
)

FEATURE_SET_ERROR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ablation_feature_set_error_analysis.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading ablation results:")
print(RESULTS_FILE)

results = pd.read_csv(RESULTS_FILE)

print("Shape:", results.shape)


print("\nLoading ablation predictions:")
print(PREDICTIONS_FILE)

predictions = pd.read_csv(PREDICTIONS_FILE)

print("Shape:", predictions.shape)


print("\nLoading final features:")
print(FEATURES_FILE)

features = pd.read_csv(FEATURES_FILE)

print("Shape:", features.shape)


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC VALIDATION")
print("=" * 60)


required_result_columns = [
    "feature_set",
    "n_features",
    "MAE",
    "RMSE",
    "R2",
]

required_prediction_columns = [
    "feature_set",
    "position",
    "actual",
    "predicted",
]

required_feature_columns = [
    "position",
    "DMS_score",
]


missing_results = [
    column
    for column in required_result_columns
    if column not in results.columns
]

missing_predictions = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

missing_features = [
    column
    for column in required_feature_columns
    if column not in features.columns
]


if missing_results:
    raise ValueError(
        "Missing ablation result columns: "
        + str(missing_results)
    )


if missing_predictions:
    raise ValueError(
        "Missing prediction columns: "
        + str(missing_predictions)
    )


if missing_features:
    raise ValueError(
        "Missing feature columns: "
        + str(missing_features)
    )


print("Required columns confirmed.")

print(
    "\nNote: original_aa and new_aa are not required here."
)

print(
    "The current final_features.csv intentionally contains"
    " encoded numerical mutation features."
)


# ============================================================
# 4. CHECK FEATURE SETS
# ============================================================

print("\nChecking feature sets...")

result_feature_sets = set(
    results["feature_set"]
)

prediction_feature_sets = set(
    predictions["feature_set"]
)


if result_feature_sets != prediction_feature_sets:

    missing_prediction_sets = (
        result_feature_sets
        - prediction_feature_sets
    )

    extra_prediction_sets = (
        prediction_feature_sets
        - result_feature_sets
    )

    raise ValueError(
        "Feature-set mismatch.\n"
        f"Missing predictions: {missing_prediction_sets}\n"
        f"Unexpected predictions: {extra_prediction_sets}"
    )


print(
    "Feature sets confirmed:",
    sorted(result_feature_sets)
)


# ============================================================
# 5. CHECK PREDICTION VALUES
# ============================================================

print("\nChecking prediction values...")

if predictions[
    ["actual", "predicted"]
].isnull().any().any():

    raise ValueError(
        "Missing actual or predicted values detected!"
    )


if not np.isfinite(
    predictions[
        ["actual", "predicted"]
    ].to_numpy()
).all():

    raise ValueError(
        "Non-finite prediction values detected!"
    )


print("No missing prediction values.")
print("No NaN or infinite prediction values.")


# ============================================================
# 6. CHECK POSITION ALIGNMENT
# ============================================================

print("\nChecking position information...")

feature_positions = set(
    features["position"].unique()
)

prediction_positions = set(
    predictions["position"].unique()
)


unknown_positions = (
    prediction_positions
    - feature_positions
)


if unknown_positions:

    raise ValueError(
        "Predictions contain unknown positions: "
        + str(sorted(unknown_positions))
    )


print(
    "All prediction positions exist in final_features.csv."
)


# ============================================================
# 7. CREATE ERROR COLUMNS
# ============================================================

print("\nCreating prediction errors...")

predictions["error"] = (
    predictions["predicted"]
    - predictions["actual"]
)

predictions["absolute_error"] = (
    predictions["error"].abs()
)

predictions["squared_error"] = (
    predictions["error"] ** 2
)

predictions["actual_abs"] = (
    predictions["actual"].abs()
)


# ============================================================
# 8. OVERALL ERROR SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("OVERALL ERROR SUMMARY")
print("=" * 60)


overall_mae = predictions[
    "absolute_error"
].mean()

overall_rmse = np.sqrt(
    predictions[
        "squared_error"
    ].mean()
)

overall_bias = predictions[
    "error"
].mean()

overall_median_absolute_error = (
    predictions[
        "absolute_error"
    ].median()
)

overall_max_absolute_error = (
    predictions[
        "absolute_error"
    ].max()
)


print(
    f"\nOverall MAE  : {overall_mae:.4f}"
)

print(
    f"Overall RMSE : {overall_rmse:.4f}"
)

print(
    f"Overall Bias : {overall_bias:.4f}"
)

print(
    "Median absolute error:",
    f"{overall_median_absolute_error:.4f}"
)

print(
    "Maximum absolute error:",
    f"{overall_max_absolute_error:.4f}"
)


# ============================================================
# 9. FEATURE-SET ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE-SET ERROR ANALYSIS")
print("=" * 60)


feature_set_rows = []


for feature_set, group in predictions.groupby(
    "feature_set"
):

    mae = group[
        "absolute_error"
    ].mean()

    rmse = np.sqrt(
        group[
            "squared_error"
        ].mean()
    )

    bias = group[
        "error"
    ].mean()

    median_error = group[
        "absolute_error"
    ].median()

    max_error = group[
        "absolute_error"
    ].max()

    n_predictions = len(group)

    feature_set_rows.append(
        {
            "feature_set": feature_set,
            "n_predictions": n_predictions,
            "MAE": mae,
            "RMSE": rmse,
            "mean_error_bias": bias,
            "median_absolute_error": median_error,
            "max_absolute_error": max_error,
        }
    )


feature_set_error = pd.DataFrame(
    feature_set_rows
)

feature_set_error = feature_set_error.sort_values(
    "MAE"
)


print(
    "\n"
    + feature_set_error.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 10. POSITION-LEVEL ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("POSITION-LEVEL ERROR ANALYSIS")
print("=" * 60)


position_error = (
    predictions
    .groupby(
        [
            "feature_set",
            "position",
        ]
    )
    .agg(
        n_predictions=(
            "actual",
            "size",
        ),
        mean_absolute_error=(
            "absolute_error",
            "mean",
        ),
        rmse=(
            "squared_error",
            lambda x: np.sqrt(
                x.mean()
            ),
        ),
        mean_error=(
            "error",
            "mean",
        ),
        max_absolute_error=(
            "absolute_error",
            "max",
        ),
    )
    .reset_index()
)


print(
    "\nNumber of position-level rows:",
    len(position_error)
)


# ============================================================
# 11. WORST POSITIONS
# ============================================================

print("\n" + "=" * 60)
print("HIGHEST-ERROR POSITIONS")
print("=" * 60)


for feature_set in sorted(
    prediction_feature_sets
):

    subset = position_error[
        position_error["feature_set"]
        == feature_set
    ].sort_values(
        "mean_absolute_error",
        ascending=False,
    ).head(10)

    print(
        f"\n{feature_set}"
    )

    print(
        subset.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# 12. WORST INDIVIDUAL PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WORST INDIVIDUAL PREDICTIONS")
print("=" * 60)


worst_predictions = (
    predictions
    .sort_values(
        "absolute_error",
        ascending=False,
    )
    .head(20)
)


print(
    "\n"
    + worst_predictions[
        [
            "feature_set",
            "position",
            "actual",
            "predicted",
            "error",
            "absolute_error",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 13. ERROR BY ACTUAL SCORE RANGE
# ============================================================

print("\n" + "=" * 60)
print("ERROR BY ACTUAL SCORE RANGE")
print("=" * 60)


def score_category(value):

    if value <= -4:
        return "Very low (<= -4)"

    if value <= -2:
        return "Low (-4 to -2)"

    if value < 0:
        return "Mild negative (-2 to 0)"

    if value == 0:
        return "Zero"

    if value <= 1:
        return "Positive (0 to 1)"

    return "High positive (> 1)"


predictions["score_range"] = (
    predictions["actual"]
    .apply(score_category)
)


score_range_error = (
    predictions
    .groupby(
        [
            "feature_set",
            "score_range",
        ],
        sort=False,
    )
    .agg(
        n_predictions=(
            "actual",
            "size",
        ),
        MAE=(
            "absolute_error",
            "mean",
        ),
        RMSE=(
            "squared_error",
            lambda x: np.sqrt(
                x.mean()
            ),
        ),
        mean_error=(
            "error",
            "mean",
        ),
    )
    .reset_index()
)


print(
    "\n"
    + score_range_error.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 14. UNDERPREDICTION / OVERPREDICTION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BIAS ANALYSIS")
print("=" * 60)


bias_rows = []


for feature_set, group in predictions.groupby(
    "feature_set"
):

    overprediction = (
        group["error"] > 0
    ).sum()

    underprediction = (
        group["error"] < 0
    ).sum()

    exact_or_zero = (
        group["error"] == 0
    ).sum()

    bias_rows.append(
        {
            "feature_set": feature_set,
            "overpredictions": overprediction,
            "underpredictions": underprediction,
            "zero_error": exact_or_zero,
            "mean_error": group["error"].mean(),
        }
    )


bias_df = pd.DataFrame(
    bias_rows
)


print(
    "\n"
    + bias_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 15. COMPARE ABLATION RESULTS WITH ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ABLATION + ERROR ANALYSIS")
print("=" * 60)


comparison = results.merge(
    feature_set_error[
        [
            "feature_set",
            "mean_error_bias",
            "median_absolute_error",
            "max_absolute_error",
        ]
    ],
    on="feature_set",
    how="left",
)


print(
    "\n"
    + comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 16. CHECK CONSISTENCY WITH ORIGINAL ABLATION METRICS
# ============================================================

print("\n" + "=" * 60)
print("METRIC CONSISTENCY CHECK")
print("=" * 60)


metric_check_rows = []


for _, row in results.iterrows():

    feature_set = row[
        "feature_set"
    ]

    subset = predictions[
        predictions["feature_set"]
        == feature_set
    ]

    calculated_mae = subset[
        "absolute_error"
    ].mean()

    calculated_rmse = np.sqrt(
        subset[
            "squared_error"
        ].mean()
    )

    mae_difference = (
        calculated_mae
        - row["MAE"]
    )

    rmse_difference = (
        calculated_rmse
        - row["RMSE"]
    )

    metric_check_rows.append(
        {
            "feature_set": feature_set,
            "reported_MAE": row["MAE"],
            "calculated_MAE": calculated_mae,
            "MAE_difference": mae_difference,
            "reported_RMSE": row["RMSE"],
            "calculated_RMSE": calculated_rmse,
            "RMSE_difference": rmse_difference,
        }
    )


metric_check = pd.DataFrame(
    metric_check_rows
)


print(
    "\n"
    + metric_check.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# 17. SAVE ERROR SUMMARY
# ============================================================

error_summary = pd.DataFrame(
    {
        "metric": [
            "overall_MAE",
            "overall_RMSE",
            "overall_bias",
            "median_absolute_error",
            "maximum_absolute_error",
            "number_of_predictions",
            "number_of_feature_sets",
        ],
        "value": [
            overall_mae,
            overall_rmse,
            overall_bias,
            overall_median_absolute_error,
            overall_max_absolute_error,
            len(predictions),
            predictions[
                "feature_set"
            ].nunique(),
        ],
    }
)


error_summary.to_csv(
    ERROR_SUMMARY_FILE,
    index=False,
)


# ============================================================
# 18. SAVE POSITION ERROR ANALYSIS
# ============================================================

position_error.to_csv(
    POSITION_ERROR_FILE,
    index=False,
)


# ============================================================
# 19. SAVE FEATURE-SET ERROR ANALYSIS
# ============================================================

feature_set_error.to_csv(
    FEATURE_SET_ERROR_FILE,
    index=False,
)


# ============================================================
# 20. SAVE DETAILED PREDICTIONS WITH ERRORS
# ============================================================

detailed_prediction_file = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_ablation_predictions_detailed.csv"
)

predictions.to_csv(
    detailed_prediction_file,
    index=False,
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)


print("\nKey findings:")

print(
    f"\nTotal predictions analyzed: "
    f"{len(predictions)}"
)

print(
    f"Feature sets analyzed: "
    f"{predictions['feature_set'].nunique()}"
)

print(
    f"Overall MAE: "
    f"{overall_mae:.4f}"
)

print(
    f"Overall RMSE: "
    f"{overall_rmse:.4f}"
)

print(
    f"Overall mean error/bias: "
    f"{overall_bias:.4f}"
)


print("\nFeature-set error summary:")

print(
    feature_set_error[
        [
            "feature_set",
            "MAE",
            "RMSE",
            "mean_error_bias",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\nSaved files:")

print(
    "\nError summary:"
)

print(
    ERROR_SUMMARY_FILE
)

print(
    "\nPosition error analysis:"
)

print(
    POSITION_ERROR_FILE
)

print(
    "\nFeature-set error analysis:"
)

print(
    FEATURE_SET_ERROR_FILE
)

print(
    "\nDetailed predictions:"
)

print(
    detailed_prediction_file
)


print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)