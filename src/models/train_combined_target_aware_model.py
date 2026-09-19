from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# ENZYOPT-AI — COMBINED TARGET-AWARE MODEL EXPERIMENT
# ============================================================

print("=" * 60)
print("ENZYOPT-AI — COMBINED TARGET-AWARE MODEL EXPERIMENT")
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

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_target_aware_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_target_aware_predictions.csv"
)

EXTREME_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "combined_target_aware_extreme_analysis.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading:")
print(DATA_FILE)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nSamples:", len(df))
print("Features:", df.shape[1] - 1)
print("Unique positions:", df["position"].nunique())


# ============================================================
# 3. VALIDATION
# ============================================================

required_columns = [
    "DMS_score",
    "position",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4. TARGET + FEATURES
# ============================================================

y = df["DMS_score"].astype(float)

groups = df["position"]

feature_columns = [
    c
    for c in df.columns
    if c not in [
        "DMS_score",
        "position",
    ]
]

X = df[feature_columns].copy()

print("\nPredictive feature matrix:")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])


if X.isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values detected in feature matrix."
    )

if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "NaN or infinite values detected in feature matrix."
    )

if not np.isfinite(y.to_numpy()).all():
    raise ValueError(
        "NaN or infinite values detected in DMS_score."
    )

print("Feature matrix validation passed.")


# ============================================================
# 5. POSITION-AWARE SPLIT
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
        X,
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

overlap = train_positions & test_positions

print("\nTraining samples:", len(train_idx))
print("Testing samples :", len(test_idx))
print("Training positions:", len(train_positions))
print("Testing positions :", len(test_positions))
print("Overlapping positions:", len(overlap))

if overlap:
    raise ValueError(
        "Position leakage detected!"
    )

print("No position overlap confirmed.")


# ============================================================
# 6. TRAIN / TEST DATA
# ============================================================

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("\nTraining target:")
print(y_train.describe())

print("\nTesting target:")
print(y_test.describe())

print(
    "\nVery low scores (<= -4):",
    int((y_test <= -4).sum()),
)

print(
    "Low scores (-4 to -2):",
    int(((y_test > -4) & (y_test <= -2)).sum()),
)

print(
    "Mild negative (-2 to 0):",
    int(((y_test > -2) & (y_test < 0)).sum()),
)

print(
    "Zero:",
    int((y_test == 0).sum()),
)

print(
    "Positive (> 0):",
    int((y_test > 0).sum()),
)


# ============================================================
# 8. SAMPLE WEIGHTS
# ============================================================

def create_sample_weights(y_values, strategy):

    y_array = np.asarray(
        y_values,
        dtype=float,
    )

    weights = np.ones(
        len(y_array),
        dtype=float,
    )

    if strategy == "baseline":
        return weights

    if strategy == "mild_extreme":
        weights[y_array <= -4] = 2.0

    elif strategy == "moderate_extreme":
        weights[y_array <= -4] = 3.0

    elif strategy == "balanced_extreme":
        weights[y_array <= -4] = 2.5
        weights[y_array > 0] = 1.5

    else:
        raise ValueError(
            f"Unknown weighting strategy: {strategy}"
        )

    return weights


# ============================================================
# 9. TARGET TRANSFORMATION
# ============================================================

def transform_target(y_values):

    y_array = np.asarray(
        y_values,
        dtype=float,
    )

    return (
        np.sign(y_array)
        * np.log1p(np.abs(y_array))
    )


def inverse_transform_target(y_values):

    y_array = np.asarray(
        y_values,
        dtype=float,
    )

    return (
        np.sign(y_array)
        * np.expm1(np.abs(y_array))
    )


# ============================================================
# 10. METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    errors = predicted - actual

    extreme_mask = actual <= -4
    positive_mask = actual > 0

    metrics = {
        "MAE": mean_absolute_error(
            actual,
            predicted,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        ),
        "R2": r2_score(
            actual,
            predicted,
        ),
        "bias": np.mean(errors),
        "median_absolute_error": np.median(
            np.abs(errors)
        ),
        "max_absolute_error": np.max(
            np.abs(errors)
        ),
    }

    if extreme_mask.any():

        metrics["extreme_mae"] = (
            mean_absolute_error(
                actual[extreme_mask],
                predicted[extreme_mask],
            )
        )

        metrics["extreme_rmse"] = np.sqrt(
            mean_squared_error(
                actual[extreme_mask],
                predicted[extreme_mask],
            )
        )

        metrics["extreme_bias"] = np.mean(
            errors[extreme_mask]
        )

    else:

        metrics["extreme_mae"] = np.nan
        metrics["extreme_rmse"] = np.nan
        metrics["extreme_bias"] = np.nan

    if positive_mask.any():

        metrics["positive_mae"] = (
            mean_absolute_error(
                actual[positive_mask],
                predicted[positive_mask],
            )
        )

    else:

        metrics["positive_mae"] = np.nan

    metrics["n_extreme"] = int(
        extreme_mask.sum()
    )

    metrics["n_positive"] = int(
        positive_mask.sum()
    )

    return metrics


# ============================================================
# 11. MODEL BUILDERS
# ============================================================

def build_model(model_name):

    if model_name == "Random Forest":

        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "Extra Trees":

        return ExtraTreesRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "HistGradientBoosting":

        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# 12. EXPERIMENTS
# ============================================================

experiments = [

    {
        "name": "Random Forest Target Transform",
        "model": "Random Forest",
        "weighting": "baseline",
    },

    {
        "name": "Random Forest Mild Extreme + Transform",
        "model": "Random Forest",
        "weighting": "mild_extreme",
    },

    {
        "name": "Random Forest Moderate Extreme + Transform",
        "model": "Random Forest",
        "weighting": "moderate_extreme",
    },

    {
        "name": "Random Forest Balanced Extreme + Transform",
        "model": "Random Forest",
        "weighting": "balanced_extreme",
    },

    {
        "name": "Extra Trees Target Transform",
        "model": "Extra Trees",
        "weighting": "baseline",
    },

    {
        "name": "Extra Trees Mild Extreme + Transform",
        "model": "Extra Trees",
        "weighting": "mild_extreme",
    },

    {
        "name": "Extra Trees Moderate Extreme + Transform",
        "model": "Extra Trees",
        "weighting": "moderate_extreme",
    },

    {
        "name": "Extra Trees Balanced Extreme + Transform",
        "model": "Extra Trees",
        "weighting": "balanced_extreme",
    },

    {
        "name": "HistGradientBoosting Target Transform",
        "model": "HistGradientBoosting",
        "weighting": "baseline",
    },

    {
        "name": "HistGradientBoosting Mild Extreme + Transform",
        "model": "HistGradientBoosting",
        "weighting": "mild_extreme",
    },

    {
        "name": "HistGradientBoosting Moderate Extreme + Transform",
        "model": "HistGradientBoosting",
        "weighting": "moderate_extreme",
    },

    {
        "name": "HistGradientBoosting Balanced Extreme + Transform",
        "model": "HistGradientBoosting",
        "weighting": "balanced_extreme",
    },

]


# ============================================================
# 13. RUN EXPERIMENTS
# ============================================================

print("\n" + "=" * 60)
print("RUNNING COMBINED EXPERIMENTS")
print("=" * 60)

results = []
prediction_rows = []
extreme_rows = []


for experiment in experiments:

    name = experiment["name"]
    model_name = experiment["model"]
    weighting = experiment["weighting"]

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print("Model:", model_name)
    print("Weighting:", weighting)
    print("Target transform: log1p_abs")

    transformed_y_train = transform_target(
        y_train
    )

    sample_weights = create_sample_weights(
        y_train,
        weighting,
    )

    print(
        "Weight range:",
        f"{sample_weights.min():.1f}",
        "to",
        f"{sample_weights.max():.1f}",
    )

    model = build_model(
        model_name
    )

    print("Training...")

    model.fit(
        X_train,
        transformed_y_train,
        sample_weight=sample_weights,
    )

    print("Training complete.")

    transformed_predictions = model.predict(
        X_test
    )

    predictions = inverse_transform_target(
        transformed_predictions
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )

    print(
        f"Bias : {metrics['bias']:.4f}"
    )

    print(
        f"Extreme MAE (<= -4): "
        f"{metrics['extreme_mae']:.4f}"
    )

    print(
        f"Positive-score MAE: "
        f"{metrics['positive_mae']:.4f}"
    )

    results.append(
        {
            "experiment": name,
            "model": model_name,
            "weighting": weighting,
            "target_transform": "log1p_abs",
            "n_features": X.shape[1],
            **metrics,
        }
    )

    for index, actual, prediction in zip(
        test_idx,
        y_test,
        predictions,
    ):

        error = prediction - actual

        prediction_rows.append(
            {
                "experiment": name,
                "model": model_name,
                "weighting": weighting,
                "target_transform": "log1p_abs",
                "position": df.iloc[index]["position"],
                "actual": actual,
                "predicted": prediction,
                "error": error,
                "absolute_error": abs(error),
            }
        )

    extreme_mask = (
        y_test.to_numpy() <= -4
    )

    extreme_indices = (
        np.asarray(test_idx)[extreme_mask]
    )

    extreme_actual = (
        y_test.to_numpy()[extreme_mask]
    )

    extreme_predictions = (
        np.asarray(predictions)[extreme_mask]
    )

    for index, actual, prediction in zip(
        extreme_indices,
        extreme_actual,
        extreme_predictions,
    ):

        error = prediction - actual

        extreme_rows.append(
            {
                "experiment": name,
                "model": model_name,
                "weighting": weighting,
                "position": df.iloc[index]["position"],
                "actual": actual,
                "predicted": prediction,
                "error": error,
                "absolute_error": abs(error),
            }
        )


# ============================================================
# 14. RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 60)
print("COMBINED TARGET-AWARE RESULTS")
print("=" * 60)

display_columns = [
    "experiment",
    "MAE",
    "RMSE",
    "R2",
    "bias",
    "extreme_mae",
    "extreme_rmse",
    "positive_mae",
]

print(
    "\n"
    + results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 15. COMPARISON WITH PREVIOUS RESULTS
# ============================================================

PREVIOUS_BASELINE_MAE = 1.2685
PREVIOUS_BASELINE_RMSE = 1.6095
PREVIOUS_BASELINE_R2 = 0.1684

TARGET_TRANSFORM_MAE = 1.1184
TARGET_TRANSFORM_RMSE = 1.5211
TARGET_TRANSFORM_R2 = 0.2572
TARGET_TRANSFORM_EXTREME_MAE = 2.8827

results_df["MAE_change_vs_previous"] = (
    results_df["MAE"]
    - PREVIOUS_BASELINE_MAE
)

results_df["RMSE_change_vs_previous"] = (
    results_df["RMSE"]
    - PREVIOUS_BASELINE_RMSE
)

results_df["R2_change_vs_previous"] = (
    results_df["R2"]
    - PREVIOUS_BASELINE_R2
)

results_df["MAE_change_vs_transform"] = (
    results_df["MAE"]
    - TARGET_TRANSFORM_MAE
)

results_df["extreme_MAE_change_vs_transform"] = (
    results_df["extreme_mae"]
    - TARGET_TRANSFORM_EXTREME_MAE
)


# ============================================================
# 16. BEST RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BEST RESULTS BY METRIC")
print("=" * 60)

best_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

best_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

best_r2 = results_df.loc[
    results_df["R2"].idxmax()
]

best_extreme = results_df.loc[
    results_df["extreme_mae"].idxmin()
]

print("\nLowest MAE:")
print(
    best_mae[
        [
            "experiment",
            "MAE",
            "RMSE",
            "R2",
            "extreme_mae",
        ]
    ].to_string()
)

print("\nLowest RMSE:")
print(
    best_rmse[
        [
            "experiment",
            "MAE",
            "RMSE",
            "R2",
            "extreme_mae",
        ]
    ].to_string()
)

print("\nHighest R²:")
print(
    best_r2[
        [
            "experiment",
            "MAE",
            "RMSE",
            "R2",
            "extreme_mae",
        ]
    ].to_string()
)

print("\nLowest extreme-score MAE:")
print(
    best_extreme[
        [
            "experiment",
            "MAE",
            "RMSE",
            "R2",
            "extreme_mae",
        ]
    ].to_string()
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

print("\nResults saved to:")
print(RESULTS_FILE)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

print("\nPredictions saved to:")
print(PREDICTIONS_FILE)


# ============================================================
# 19. SAVE EXTREME ANALYSIS
# ============================================================

extreme_df = pd.DataFrame(
    extreme_rows
)

extreme_df = extreme_df.sort_values(
    "absolute_error",
    ascending=False,
)

extreme_df.to_csv(
    EXTREME_FILE,
    index=False,
)

print("\nExtreme-score analysis saved to:")
print(EXTREME_FILE)


# ============================================================
# 20. WORST EXTREME PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WORST EXTREME-SCORE PREDICTIONS")
print("=" * 60)

print(
    "\n"
    + extreme_df.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COMBINED TARGET-AWARE EXPERIMENT COMPLETE")
print("=" * 60)

print("\nExperiments evaluated:")
print(len(results_df))

print("\nPrevious all-feature baseline:")
print("MAE :", f"{PREVIOUS_BASELINE_MAE:.4f}")
print("RMSE:", f"{PREVIOUS_BASELINE_RMSE:.4f}")
print("R²  :", f"{PREVIOUS_BASELINE_R2:.4f}")

print("\nPrevious target-transform baseline:")
print("MAE :", f"{TARGET_TRANSFORM_MAE:.4f}")
print("RMSE:", f"{TARGET_TRANSFORM_RMSE:.4f}")
print("R²  :", f"{TARGET_TRANSFORM_R2:.4f}")
print(
    "Extreme MAE:",
    f"{TARGET_TRANSFORM_EXTREME_MAE:.4f}",
)

print("\nBest overall MAE:")
print(best_mae["experiment"])
print(f"MAE: {best_mae['MAE']:.4f}")

print("\nBest overall RMSE:")
print(best_rmse["experiment"])
print(f"RMSE: {best_rmse['RMSE']:.4f}")

print("\nBest overall R²:")
print(best_r2["experiment"])
print(f"R²: {best_r2['R2']:.4f}")

print("\nBest extreme-score MAE:")
print(best_extreme["experiment"])
print(
    f"Extreme MAE: "
    f"{best_extreme['extreme_mae']:.4f}"
)

print("\nAll experiments used the same")
print("position-aware train/test split.")

print("Extreme observations were retained.")

print("Target transformation was applied")
print("only to the training target.")

print("Sample weighting was applied")
print("only during training.")

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)