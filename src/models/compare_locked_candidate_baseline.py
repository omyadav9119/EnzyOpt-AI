from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# ENZYOPT-AI — LOCKED CANDIDATE VS BASELINE CV
# ============================================================
# Purpose:
#   Compare the locked HGB candidate against simple
#   position-independent baselines under the SAME
#   repeated position-aware CV protocol.
#
# IMPORTANT:
#   - Candidate model is NOT changed.
#   - Original fixed holdout remains untouched.
#   - No tuning is performed.
#
# Baselines:
#   1. Training-fold mean target
#   2. Training-fold median target
#
# This tells us whether the model transfers useful information
# to unseen positions beyond a constant prediction.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.csv"
)

OUT_DIR = PROJECT_ROOT / "data" / "processed"

FOLD_RESULTS_FILE = (
    OUT_DIR
    / "locked_candidate_vs_baseline_cv.csv"
)

SUMMARY_FILE = (
    OUT_DIR
    / "locked_candidate_vs_baseline_summary.csv"
)

POSITION_FILE = (
    OUT_DIR
    / "locked_candidate_vs_baseline_position.csv"
)


# ============================================================
# 2. LOCKED MODEL
# ============================================================

MODEL_PARAMS = dict(
    max_iter=300,
    learning_rate=0.04,
    max_leaf_nodes=15,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=42,
)


def transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def inverse_transform_target(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.expm1(np.abs(y))


def sample_weights(y):
    y = np.asarray(y, dtype=float)

    weights = np.ones(len(y), dtype=float)
    weights[y <= -4] = 2.0

    return weights


def build_model():
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


# ============================================================
# 3. METRICS
# ============================================================

def calculate_metrics(actual, pred):

    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    error = pred - actual

    extreme = actual <= -4
    positive = actual > 0

    result = {
        "n": len(actual),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred),
        "bias": np.mean(error),
        "median_abs_error": np.median(np.abs(error)),
        "max_abs_error": np.max(np.abs(error)),
        "extreme_n": int(np.sum(extreme)),
        "positive_n": int(np.sum(positive)),
    }

    if np.any(extreme):
        result["extreme_MAE"] = mean_absolute_error(
            actual[extreme],
            pred[extreme],
        )
    else:
        result["extreme_MAE"] = np.nan

    if np.any(positive):
        result["positive_MAE"] = mean_absolute_error(
            actual[positive],
            pred[positive],
        )
    else:
        result["positive_MAE"] = np.nan

    return result


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("ENZYOPT-AI — LOCKED CANDIDATE VS BASELINE CV")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"\nDataset shape: {df.shape}")

target_col = "DMS_score"
group_col = "position"

feature_cols = [
    c
    for c in df.columns
    if c not in [target_col, group_col]
]

X = df[feature_cols].copy()
y = df[target_col].astype(float).copy()
groups = df[group_col].copy()


# ============================================================
# 5. REPRODUCE ORIGINAL FIXED HOLDOUT
# ============================================================
# This is only used to preserve the exact original split.
# It is NOT used during CV.
# ============================================================

original_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, holdout_idx = next(
    original_splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()
groups_train = groups.iloc[train_idx].copy()

X_holdout = X.iloc[holdout_idx].copy()
y_holdout = y.iloc[holdout_idx].copy()

print("\nOriginal split:")
print(f"  Training samples: {len(train_idx)}")
print(f"  Holdout samples:  {len(holdout_idx)}")
print(
    f"  Training positions: "
    f"{groups_train.nunique()}"
)
print(
    f"  Holdout positions: "
    f"{groups.iloc[holdout_idx].nunique()}"
)


# ============================================================
# 6. REPEATED POSITION-AWARE CV
# ============================================================

N_REPEATS = 3
N_FOLDS = 5

all_results = []
all_position_results = []

for repeat in range(1, N_REPEATS + 1):

    print("\n" + "-" * 70)
    print(f"REPEAT {repeat}/{N_REPEATS}")
    print("-" * 70)

    cv = GroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=1000 + repeat,
    )

    for fold, (cv_train_idx, cv_val_idx) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups=groups_train,
        ),
        start=1,
    ):

        X_tr = X_train.iloc[cv_train_idx]
        y_tr = y_train.iloc[cv_train_idx]

        X_val = X_train.iloc[cv_val_idx]
        y_val = y_train.iloc[cv_val_idx]

        groups_val = groups_train.iloc[cv_val_idx]

        # ----------------------------------------------------
        # LOCKED HGB MODEL
        # ----------------------------------------------------

        model = build_model()

        y_tr_transformed = transform_target(y_tr)

        model.fit(
            X_tr,
            y_tr_transformed,
            sample_weight=sample_weights(y_tr),
        )

        pred_model = inverse_transform_target(
            model.predict(X_val)
        )

        # ----------------------------------------------------
        # BASELINE 1: TRAINING-FOLD MEAN
        # ----------------------------------------------------

        mean_prediction = float(y_tr.mean())

        pred_mean = np.full(
            len(y_val),
            mean_prediction,
            dtype=float,
        )

        # ----------------------------------------------------
        # BASELINE 2: TRAINING-FOLD MEDIAN
        # ----------------------------------------------------

        median_prediction = float(y_tr.median())

        pred_median = np.full(
            len(y_val),
            median_prediction,
            dtype=float,
        )

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        model_metrics = calculate_metrics(
            y_val,
            pred_model,
        )

        mean_metrics = calculate_metrics(
            y_val,
            pred_mean,
        )

        median_metrics = calculate_metrics(
            y_val,
            pred_median,
        )

        # ----------------------------------------------------
        # STORE FOLD RESULTS
        # ----------------------------------------------------

        for method, metrics in [
            ("locked_hgb", model_metrics),
            ("mean_baseline", mean_metrics),
            ("median_baseline", median_metrics),
        ]:

            row = {
                "repeat": repeat,
                "fold": fold,
                "method": method,
                "mean_baseline_value": mean_prediction,
                "median_baseline_value": median_prediction,
                **metrics,
            }

            all_results.append(row)

        # ----------------------------------------------------
        # IMPROVEMENT OVER BASELINES
        # ----------------------------------------------------

        model_mae = model_metrics["MAE"]
        model_rmse = model_metrics["RMSE"]

        mean_mae = mean_metrics["MAE"]
        mean_rmse = mean_metrics["RMSE"]

        median_mae = median_metrics["MAE"]
        median_rmse = median_metrics["RMSE"]

        print(
            f"Fold {fold}: "
            f"HGB MAE={model_mae:.4f} | "
            f"Mean MAE={mean_mae:.4f} | "
            f"Median MAE={median_mae:.4f}"
        )

        print(
            f"         "
            f"HGB RMSE={model_rmse:.4f} | "
            f"Mean RMSE={mean_rmse:.4f} | "
            f"Median RMSE={median_rmse:.4f}"
        )

        print(
            f"         "
            f"MAE improvement vs mean="
            f"{mean_mae - model_mae:+.4f}"
        )

        # ----------------------------------------------------
        # POSITION-LEVEL COMPARISON
        # ----------------------------------------------------

        validation_frame = pd.DataFrame(
            {
                "position": groups_val.values,
                "actual": y_val.values,
                "hgb_pred": pred_model,
                "mean_pred": pred_mean,
                "median_pred": pred_median,
            }
        )

        for position, pos_df in validation_frame.groupby(
            "position"
        ):

            actual = pos_df["actual"].values

            hgb_pred = pos_df["hgb_pred"].values
            mean_pred = pos_df["mean_pred"].values
            median_pred = pos_df["median_pred"].values

            hgb_mae = mean_absolute_error(
                actual,
                hgb_pred,
            )

            mean_mae_pos = mean_absolute_error(
                actual,
                mean_pred,
            )

            median_mae_pos = mean_absolute_error(
                actual,
                median_pred,
            )

            all_position_results.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "position": position,
                    "n": len(pos_df),
                    "actual_mean": np.mean(actual),
                    "actual_std": (
                        np.std(actual)
                        if len(actual) > 1
                        else 0.0
                    ),
                    "hgb_MAE": hgb_mae,
                    "mean_baseline_MAE": mean_mae_pos,
                    "median_baseline_MAE": median_mae_pos,
                    "hgb_minus_mean_MAE": (
                        hgb_mae - mean_mae_pos
                    ),
                    "hgb_minus_median_MAE": (
                        hgb_mae - median_mae_pos
                    ),
                    "hgb_beats_mean": (
                        hgb_mae < mean_mae_pos
                    ),
                    "hgb_beats_median": (
                        hgb_mae < median_mae_pos
                    ),
                }
            )


# ============================================================
# 7. SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

position_df = pd.DataFrame(
    all_position_results
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    FOLD_RESULTS_FILE,
    index=False,
)

position_df.to_csv(
    POSITION_FILE,
    index=False,
)


# ============================================================
# 8. SUMMARY
# ============================================================

summary_rows = []

metric_columns = [
    "MAE",
    "RMSE",
    "R2",
    "bias",
    "median_abs_error",
    "max_abs_error",
    "extreme_MAE",
    "positive_MAE",
]

for method in [
    "locked_hgb",
    "mean_baseline",
    "median_baseline",
]:

    method_df = results_df[
        results_df["method"] == method
    ]

    row = {
        "method": method,
        "n_folds": len(method_df),
    }

    for metric in metric_columns:

        row[f"{metric}_mean"] = (
            method_df[metric].mean()
        )

        row[f"{metric}_std"] = (
            method_df[metric].std()
        )

        row[f"{metric}_min"] = (
            method_df[metric].min()
        )

        row[f"{metric}_max"] = (
            method_df[metric].max()
        )

        row[f"{metric}_median"] = (
            method_df[metric].median()
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# 9. REPEAT-LEVEL SUMMARY
# ============================================================

repeat_summary = (
    results_df
    .groupby(["repeat", "method"])
    [["MAE", "RMSE", "R2", "bias",
      "extreme_MAE", "positive_MAE"]]
    .mean()
    .reset_index()
)


# ============================================================
# 10. POSITION-LEVEL RESULTS
# ============================================================

position_summary = (
    position_df
    .groupby("position")
    [
        [
            "n",
            "actual_mean",
            "hgb_MAE",
            "mean_baseline_MAE",
            "median_baseline_MAE",
            "hgb_minus_mean_MAE",
            "hgb_minus_median_MAE",
            "hgb_beats_mean",
            "hgb_beats_median",
        ]
    ]
    .agg(
        {
            "n": "mean",
            "actual_mean": "mean",
            "hgb_MAE": "mean",
            "mean_baseline_MAE": "mean",
            "median_baseline_MAE": "mean",
            "hgb_minus_mean_MAE": "mean",
            "hgb_minus_median_MAE": "mean",
            "hgb_beats_mean": "mean",
            "hgb_beats_median": "mean",
        }
    )
    .reset_index()
)


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL CV SUMMARY")
print("=" * 70)

display_columns = [
    "method",
    "MAE_mean",
    "MAE_std",
    "RMSE_mean",
    "RMSE_std",
    "R2_mean",
    "R2_std",
    "bias_mean",
    "extreme_MAE_mean",
    "positive_MAE_mean",
]

print(
    summary_df[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 70)
print("REPEAT-LEVEL SUMMARY")
print("=" * 70)

print(
    repeat_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 12. DIRECT MODEL VS BASELINE COMPARISON
# ============================================================

hgb_df = results_df[
    results_df["method"] == "locked_hgb"
].copy()

mean_df = results_df[
    results_df["method"] == "mean_baseline"
].copy()

median_df = results_df[
    results_df["method"] == "median_baseline"
].copy()


hgb_mae = hgb_df["MAE"].values
mean_mae = mean_df["MAE"].values
median_mae = median_df["MAE"].values

hgb_rmse = hgb_df["RMSE"].values
mean_rmse = mean_df["RMSE"].values
median_rmse = median_df["RMSE"].values


print("\n" + "=" * 70)
print("MODEL VS BASELINE")
print("=" * 70)

print(
    f"\nMean-baseline MAE improvement: "
    f"{np.mean(mean_mae - hgb_mae):+.4f}"
)

print(
    f"Median-baseline MAE improvement: "
    f"{np.mean(median_mae - hgb_mae):+.4f}"
)

print(
    f"\nMean-baseline RMSE improvement: "
    f"{np.mean(mean_rmse - hgb_rmse):+.4f}"
)

print(
    f"Median-baseline RMSE improvement: "
    f"{np.mean(median_rmse - hgb_rmse):+.4f}"
)


# How often does HGB beat the baselines?

hgb_beats_mean_mae = np.sum(
    hgb_mae < mean_mae
)

hgb_beats_median_mae = np.sum(
    hgb_mae < median_mae
)

total_folds = len(hgb_mae)

print(
    f"\nHGB beats mean baseline on MAE: "
    f"{hgb_beats_mean_mae}/{total_folds} "
    f"({100*hgb_beats_mean_mae/total_folds:.1f}%)"
)

print(
    f"HGB beats median baseline on MAE: "
    f"{hgb_beats_median_mae}/{total_folds} "
    f"({100*hgb_beats_median_mae/total_folds:.1f}%)"
)


# ============================================================
# 13. POSITION-LEVEL COMPARISON
# ============================================================

position_counts = {
    "positions": len(position_summary),
    "hgb_beats_mean": (
        position_summary["hgb_beats_mean"].mean()
    ),
    "hgb_beats_median": (
        position_summary["hgb_beats_median"].mean()
    ),
}

print("\n" + "=" * 70)
print("POSITION-LEVEL COMPARISON")
print("=" * 70)

print(
    f"\nUnique validation positions evaluated: "
    f"{position_counts['positions']:.0f}"
)

print(
    f"Fraction of position/fold cases where HGB "
    f"beats mean baseline: "
    f"{100*position_counts['hgb_beats_mean']:.1f}%"
)

print(
    f"Fraction of position/fold cases where HGB "
    f"beats median baseline: "
    f"{100*position_counts['hgb_beats_median']:.1f}%"
)


print("\nHardest positions where HGB loses most heavily to mean baseline:")

hard_positions = (
    position_summary
    .sort_values(
        "hgb_minus_mean_MAE",
        ascending=False,
    )
    .head(15)
)

print(
    hard_positions[
        [
            "position",
            "n",
            "hgb_MAE",
            "mean_baseline_MAE",
            "hgb_minus_mean_MAE",
            "actual_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\nPositions where HGB improves most over mean baseline:")

easy_positions = (
    position_summary
    .sort_values(
        "hgb_minus_mean_MAE",
        ascending=True,
    )
    .head(15)
)

print(
    easy_positions[
        [
            "position",
            "n",
            "hgb_MAE",
            "mean_baseline_MAE",
            "hgb_minus_mean_MAE",
            "actual_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# 14. FINAL INTERPRETATION
# ============================================================

overall_hgb_mae = hgb_df["MAE"].mean()
overall_mean_mae = mean_df["MAE"].mean()
overall_median_mae = median_df["MAE"].mean()

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if overall_hgb_mae < overall_mean_mae:
    print(
        "\nThe locked HGB has lower mean MAE than the "
        "global-mean baseline under repeated position-aware CV."
    )
else:
    print(
        "\nThe locked HGB does NOT have lower mean MAE than "
        "the global-mean baseline under repeated "
        "position-aware CV."
    )

if overall_hgb_mae < overall_median_mae:
    print(
        "It also has lower mean MAE than the "
        "global-median baseline."
    )
else:
    print(
        "It does NOT have lower mean MAE than the "
        "global-median baseline."
    )

print(
    "\nThis comparison does not change or tune the model."
)

print(
    "It tests whether the current feature set provides "
    "transferable information for unseen positions."
)

print("\nSaved:")
print(f"  {FOLD_RESULTS_FILE}")
print(f"  {SUMMARY_FILE}")
print(f"  {POSITION_FILE}")