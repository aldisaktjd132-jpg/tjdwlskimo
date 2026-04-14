from pathlib import Path
import json
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


BASE = Path("/content")
TENSOR_DIR = BASE / "convlstm_monthly_tensors"
MODEL_DIR = BASE / "convlstm_monthly_model"


def clip_probs(probs):
    return np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)


def logit_clip(probs):
    probs = clip_probs(probs)
    return np.log(probs / (1.0 - probs))


def safe_div(a, b):
    return a / b if b else 0.0


def safe_auc(metric_fn, targets, probs):
    try:
        return float(metric_fn(targets, probs))
    except ValueError:
        return float("nan")


def compute_metrics(targets, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_prob_metrics(targets, probs):
    probs = clip_probs(probs)
    targets = np.asarray(targets, dtype=int)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask = (probs >= left) & (probs < right if i < len(bins) - 2 else probs <= right)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(targets[mask].mean() - probs[mask].mean())
    prevalence = float(targets.mean())
    baseline_probs = np.full_like(probs, prevalence, dtype=np.float64)
    return {
        "brier": float(brier_score_loss(targets, probs)),
        "baseline_brier": float(brier_score_loss(targets, baseline_probs)),
        "pr_auc": safe_auc(average_precision_score, targets, probs),
        "roc_auc": safe_auc(roc_auc_score, targets, probs),
        "ece": float(ece),
        "prevalence": prevalence,
    }


def build_threshold_search_table(targets, probs):
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)
    rows = [compute_metrics(targets, probs, float(threshold)) for threshold in thresholds]
    return pd.DataFrame(rows)


def select_threshold(validation_table, recall_floor=0.70):
    eligible = validation_table[validation_table["recall"] >= recall_floor].copy()
    used_fallback = eligible.empty
    candidate_table = validation_table if used_fallback else eligible
    selected = (
        candidate_table
        .sort_values(["f1", "precision", "threshold"], ascending=[False, False, True])
        .iloc[0]
        .to_dict()
    )
    return selected, used_fallback


def build_calibration_table(targets, probs, n_bins=10):
    frac_pos, mean_pred = calibration_curve(targets, clip_probs(probs), n_bins=n_bins, strategy="quantile")
    clipped = clip_probs(probs)
    quantiles = np.quantile(clipped, np.linspace(0, 1, n_bins + 1))
    rows = []
    for idx in range(n_bins):
        left = float(quantiles[idx])
        right = float(quantiles[idx + 1])
        if idx == n_bins - 1:
            mask = (clipped >= left) & (clipped <= right)
        else:
            mask = (clipped >= left) & (clipped < right)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": idx + 1,
                "bin_left": left,
                "bin_right": right,
                "count": int(mask.sum()),
                "mean_predicted_probability": float(clipped[mask].mean()),
                "observed_positive_rate": float(np.asarray(targets)[mask].mean()),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty and len(frac_pos) == len(table) and len(mean_pred) == len(table):
        table["calibration_curve_mean_pred"] = mean_pred
        table["calibration_curve_frac_pos"] = frac_pos
    return table


def compute_calibration_slope_intercept(targets, probs):
    targets = np.asarray(targets, dtype=int)
    if np.unique(targets).size < 2:
        return float("nan"), float("nan")
    x = logit_clip(probs).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(x, targets)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def apply_monthly_calibrator(method_name, model_obj, probs):
    arr = clip_probs(np.asarray(probs, dtype=float))
    flat = arr.reshape(-1)
    if method_name == "identity":
        out = flat
    elif method_name == "platt_logit_1d":
        out = model_obj.predict_proba(logit_clip(flat).reshape(-1, 1))[:, 1]
    elif method_name == "isotonic_1d":
        out = model_obj.predict(flat)
    else:
        raise ValueError(f"unknown monthly calibration method: {method_name}")
    return clip_probs(out).reshape(arr.shape)


def select_monthly_calibration(valid_targets, valid_probs, test_probs):
    rows = []
    models = {}

    def add_candidate(method_name, valid_pred, test_pred, model_obj=None):
        metrics = compute_prob_metrics(valid_targets, valid_pred)
        rows.append(
            {
                "method": method_name,
                "valid_brier": metrics["brier"],
                "valid_baseline_brier": metrics["baseline_brier"],
                "valid_pr_auc": metrics["pr_auc"],
                "valid_roc_auc": metrics["roc_auc"],
                "valid_ece": metrics["ece"],
            }
        )
        models[method_name] = {
            "model": model_obj,
            "valid_probs": clip_probs(valid_pred),
            "test_probs": clip_probs(test_pred),
        }

    add_candidate("identity", valid_probs, test_probs, None)

    if np.unique(valid_targets).size >= 2:
        platt_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt_model.fit(logit_clip(valid_probs).reshape(-1, 1), valid_targets)
        add_candidate(
            "platt_logit_1d",
            platt_model.predict_proba(logit_clip(valid_probs).reshape(-1, 1))[:, 1],
            platt_model.predict_proba(logit_clip(test_probs).reshape(-1, 1))[:, 1],
            platt_model,
        )

        iso_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_model.fit(clip_probs(valid_probs), valid_targets)
        add_candidate(
            "isotonic_1d",
            iso_model.predict(clip_probs(valid_probs)),
            iso_model.predict(clip_probs(test_probs)),
            iso_model,
        )

    table = pd.DataFrame(rows)
    identity_row = table.loc[table["method"] == "identity"].iloc[0]
    pr_auc_floor = float(identity_row["valid_pr_auc"]) - 0.01
    shortlisted = table[table["valid_pr_auc"] >= pr_auc_floor].copy()
    if shortlisted.empty:
        shortlisted = table.copy()
    improved = shortlisted[
        (shortlisted["valid_brier"] <= float(identity_row["valid_brier"]) + 1e-9)
        | (shortlisted["valid_ece"] <= float(identity_row["valid_ece"]) + 1e-9)
    ].copy()
    if not improved.empty:
        shortlisted = improved
    table = table.sort_values(
        ["valid_brier", "valid_ece", "valid_pr_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best_method = str(
        shortlisted.sort_values(
            ["valid_brier", "valid_ece", "valid_pr_auc"],
            ascending=[True, True, False],
        ).iloc[0]["method"]
    )
    selected = models[best_method]
    return best_method, selected["model"], selected["valid_probs"], selected["test_probs"], table


def select_annual_calibration(annual_valid_df, annual_future_df):
    if annual_valid_df.empty or annual_valid_df["annual_any_actual"].nunique() < 2:
        annual_future_df = annual_future_df.copy()
        annual_future_df["annual_calibrated_probability"] = clip_probs(annual_future_df["annual_raw_probability"])
        table = pd.DataFrame(
            [{"method": "identity", "valid_brier": np.nan, "valid_baseline_brier": np.nan, "valid_pr_auc": np.nan, "valid_roc_auc": np.nan, "valid_ece": np.nan}]
        )
        return "identity", None, annual_future_df["annual_calibrated_probability"].to_numpy(dtype=float), table

    y_valid = annual_valid_df["annual_any_actual"].astype(int).to_numpy()
    raw_valid = clip_probs(annual_valid_df["annual_raw_probability"].to_numpy(dtype=float))
    raw_future = clip_probs(annual_future_df["annual_raw_probability"].to_numpy(dtype=float))
    risk_valid = annual_valid_df["annual_risk_score"].astype(float).to_numpy()
    risk_future = annual_future_df["annual_risk_score"].astype(float).to_numpy()

    rows = []
    models = {}

    def add_candidate(method_name, valid_pred, future_pred, model_obj=None):
        metrics = compute_prob_metrics(y_valid, valid_pred)
        rows.append(
            {
                "method": method_name,
                "valid_brier": metrics["brier"],
                "valid_baseline_brier": metrics["baseline_brier"],
                "valid_pr_auc": metrics["pr_auc"],
                "valid_roc_auc": metrics["roc_auc"],
                "valid_ece": metrics["ece"],
            }
        )
        models[method_name] = {"model": model_obj, "future_probs": clip_probs(future_pred)}

    add_candidate("identity", raw_valid, raw_future, None)

    iso_model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_model.fit(raw_valid, y_valid)
    add_candidate("isotonic_1d", iso_model.predict(raw_valid), iso_model.predict(raw_future), iso_model)

    platt1_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt1_model.fit(logit_clip(raw_valid).reshape(-1, 1), y_valid)
    add_candidate(
        "platt_logit_1d",
        platt1_model.predict_proba(logit_clip(raw_valid).reshape(-1, 1))[:, 1],
        platt1_model.predict_proba(logit_clip(raw_future).reshape(-1, 1))[:, 1],
        platt1_model,
    )

    platt2_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    x_valid_2d = np.column_stack([logit_clip(raw_valid), risk_valid])
    x_future_2d = np.column_stack([logit_clip(raw_future), risk_future])
    platt2_model.fit(x_valid_2d, y_valid)
    add_candidate(
        "platt_logit_2d",
        platt2_model.predict_proba(x_valid_2d)[:, 1],
        platt2_model.predict_proba(x_future_2d)[:, 1],
        platt2_model,
    )

    table = pd.DataFrame(rows)
    identity_row = table.loc[table["method"] == "identity"].iloc[0]
    pr_auc_floor = float(identity_row["valid_pr_auc"]) - 0.01 if np.isfinite(identity_row["valid_pr_auc"]) else -np.inf
    shortlisted = table[table["valid_pr_auc"] >= pr_auc_floor].copy()
    if shortlisted.empty:
        shortlisted = table.copy()
    improved = shortlisted[
        (shortlisted["valid_brier"] <= float(identity_row["valid_brier"]) + 1e-9)
        | (shortlisted["valid_ece"] <= float(identity_row["valid_ece"]) + 1e-9)
    ].copy()
    if not improved.empty:
        shortlisted = improved
    table = table.sort_values(
        ["valid_brier", "valid_ece", "valid_pr_auc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best_method = str(
        shortlisted.sort_values(
            ["valid_brier", "valid_ece", "valid_pr_auc"],
            ascending=[True, True, False],
        ).iloc[0]["method"]
    )
    selected = models[best_method]
    return best_method, selected["model"], selected["future_probs"], table


def build_annual_from_monthly_samples(meta_df, probs_4d, targets_4d, mapping_df):
    frames = []
    mapping_df = mapping_df[["grid_id", "row_idx", "col_idx"]].copy()
    for sample_idx, row in meta_df.reset_index(drop=True).iterrows():
        target_month = str(row["target_start_month"])
        pred_year = int(target_month[:4])
        block = mapping_df.copy()
        block["pred_year"] = pred_year
        block["pred_month"] = target_month
        prob_map = clip_probs(np.asarray(probs_4d[sample_idx, 0], dtype=float))
        target_map = np.asarray(targets_4d[sample_idx, 0], dtype=float)
        block["monthly_prob"] = prob_map[block["row_idx"], block["col_idx"]]
        block["monthly_actual"] = target_map[block["row_idx"], block["col_idx"]]
        frames.append(block)

    monthly_df = pd.concat(frames, ignore_index=True)
    annual_df = (
        monthly_df
        .groupby(["pred_year", "grid_id", "row_idx", "col_idx"], as_index=False)
        .agg(
            annual_expected_monthly_positive_sum=("monthly_prob", "sum"),
            annual_raw_probability=(
                "monthly_prob",
                lambda s: float(1.0 - np.prod(1.0 - clip_probs(s.to_numpy(dtype=float)))),
            ),
            annual_any_actual=("monthly_actual", "max"),
            month_count=("monthly_prob", "size"),
        )
    )
    return annual_df


def plot_pr_curve(targets, probs, output_path):
    precision, recall, _ = precision_recall_curve(targets, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR curve", color="#1d4ed8")
    plt.axhline(np.mean(targets), linestyle="--", color="#dc2626", label="Prevalence baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Test Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curve(targets, probs, output_path):
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve", color="#1d4ed8")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_calibration_curve(calibration_table, output_path):
    plt.figure(figsize=(6, 5))
    plt.plot(
        calibration_table["mean_predicted_probability"],
        calibration_table["observed_positive_rate"],
        marker="o",
        color="#1d4ed8",
        label="Model",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#6b7280", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Test Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_metrics(validation_table, selected_threshold, output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(validation_table["threshold"], validation_table["precision"], label="Precision", color="#2563eb")
    plt.plot(validation_table["threshold"], validation_table["recall"], label="Recall", color="#16a34a")
    plt.plot(validation_table["threshold"], validation_table["f1"], label="F1", color="#dc2626")
    plt.axvline(selected_threshold, linestyle="--", color="#111827", label=f"Selected = {selected_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Validation Threshold Search")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    validation_threshold_csv = MODEL_DIR / "monthly_validation_threshold_search.csv"
    test_metrics_csv = MODEL_DIR / "monthly_test_final_metrics.csv"
    calibration_table_csv = MODEL_DIR / "monthly_test_calibration_curve.csv"
    monthly_calibration_csv = MODEL_DIR / "monthly_calibration_method_comparison.csv"
    annual_calibration_csv = MODEL_DIR / "annual_calibration_method_comparison.csv"
    pr_curve_png = MODEL_DIR / "monthly_test_pr_curve.png"
    roc_curve_png = MODEL_DIR / "monthly_test_roc_curve.png"
    calibration_curve_png = MODEL_DIR / "monthly_test_calibration_curve.png"
    threshold_metrics_png = MODEL_DIR / "monthly_validation_threshold_metrics.png"
    eval_summary_json = MODEL_DIR / "convlstm_monthly_eval_summary.json"

    required = [
        MODEL_DIR / "best_convlstm_monthly.pt",
        MODEL_DIR / "temperature_scaling.json",
        TENSOR_DIR / "X_valid.npy",
        TENSOR_DIR / "y_valid.npy",
        TENSOR_DIR / "X_test.npy",
        TENSOR_DIR / "y_test.npy",
        TENSOR_DIR / "meta_train.csv",
        TENSOR_DIR / "meta_valid.csv",
        TENSOR_DIR / "meta_test.csv",
        BASE / "monthly_prediction_probabilities_2025_2027.csv",
        BASE / "grid_id_to_row_col_mapping_rebuilt.csv",
        BASE / "train_convlstm_monthly_colab_gpu.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다.\n" + "\n".join(missing))

    train_module = runpy.run_path(str(BASE / "train_convlstm_monthly_colab_gpu.py"))
    ConvLSTMForecaster = train_module["ConvLSTMForecaster"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_DIR / "best_convlstm_monthly.pt", map_location=device)
    model = ConvLSTMForecaster(
        input_dim=ckpt["input_dim"],
        output_horizon=ckpt.get("output_horizon", 1),
        hidden_dim=ckpt["hidden_dim"],
        kernel_size=ckpt["kernel_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(MODEL_DIR / "temperature_scaling.json", "r", encoding="utf-8") as f:
        temperature_value = float(json.load(f)["temperature"])

    X_valid = torch.tensor(np.load(TENSOR_DIR / "X_valid.npy"), dtype=torch.float32, device=device)
    y_valid = torch.tensor(np.load(TENSOR_DIR / "y_valid.npy"), dtype=torch.float32, device=device)
    X_test = torch.tensor(np.load(TENSOR_DIR / "X_test.npy"), dtype=torch.float32, device=device)
    y_test = torch.tensor(np.load(TENSOR_DIR / "y_test.npy"), dtype=torch.float32, device=device)

    meta_train = pd.read_csv(TENSOR_DIR / "meta_train.csv")
    meta_valid = pd.read_csv(TENSOR_DIR / "meta_valid.csv")
    meta_test = pd.read_csv(TENSOR_DIR / "meta_test.csv")
    monthly_future = pd.read_csv(BASE / "monthly_prediction_probabilities_2025_2027.csv")
    mapping = pd.read_csv(BASE / "grid_id_to_row_col_mapping_rebuilt.csv")

    with torch.no_grad():
        valid_logits = model(X_valid) / temperature_value
        test_logits = model(X_test) / temperature_value

    valid_probs = torch.sigmoid(valid_logits).cpu().numpy()
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
    valid_targets = y_valid.cpu().numpy()
    test_targets = y_test.cpu().numpy()

    valid_probs_flat = valid_probs.reshape(-1)
    test_probs_flat = test_probs.reshape(-1)
    valid_targets_flat = valid_targets.reshape(-1).astype(int)
    test_targets_flat = test_targets.reshape(-1).astype(int)

    monthly_calibration_method, monthly_calibration_model, valid_probs_cal, test_probs_cal, monthly_calibration_table = select_monthly_calibration(
        valid_targets_flat,
        valid_probs_flat,
        test_probs_flat,
    )
    monthly_calibration_table.to_csv(monthly_calibration_csv, index=False, encoding="utf-8-sig")

    raw_test_prob_metrics = compute_prob_metrics(test_targets_flat, test_probs_flat)
    calibrated_test_prob_metrics = compute_prob_metrics(test_targets_flat, test_probs_cal)

    validation_threshold_table = build_threshold_search_table(valid_targets_flat, valid_probs_cal)
    validation_threshold_table.to_csv(validation_threshold_csv, index=False, encoding="utf-8-sig")

    selection_recall_floor = 0.70
    best_valid, used_fallback = select_threshold(validation_threshold_table, recall_floor=selection_recall_floor)
    selected_threshold = float(best_valid["threshold"])

    test_metrics = compute_metrics(test_targets_flat, test_probs_cal, selected_threshold)
    calibration_table = build_calibration_table(test_targets_flat, test_probs_cal, n_bins=10)
    calibration_table.to_csv(calibration_table_csv, index=False, encoding="utf-8-sig")
    calibration_intercept, calibration_slope = compute_calibration_slope_intercept(test_targets_flat, test_probs_cal)

    monthly_future = monthly_future.copy()
    monthly_future["monthly_probability_after_monthly_calibration"] = apply_monthly_calibrator(
        monthly_calibration_method,
        monthly_calibration_model,
        monthly_future["calibrated_probability_next_month"].to_numpy(dtype=float),
    )
    monthly_future.to_csv(BASE / "monthly_prediction_probabilities_recalibrated_2025_2027.csv", index=False, encoding="utf-8-sig")

    annual_raw_future = (
        monthly_future
        .groupby(["pred_year", "grid_id", "row_idx", "col_idx"], as_index=False)
        .agg(
            annual_expected_monthly_positive_sum=("monthly_probability_after_monthly_calibration", "sum"),
            annual_raw_probability=(
                "monthly_probability_after_monthly_calibration",
                lambda s: float(1.0 - np.prod(1.0 - clip_probs(s.to_numpy(dtype=float)))),
            ),
        )
    )
    annual_raw_future["annual_risk_score"] = annual_raw_future["annual_expected_monthly_positive_sum"]
    annual_raw_future.to_csv(BASE / "annual_prediction_raw_rebuilt_2025_2027.csv", index=False, encoding="utf-8-sig")

    valid_probs_cal_4d = apply_monthly_calibrator(monthly_calibration_method, monthly_calibration_model, valid_probs)
    annual_valid = build_annual_from_monthly_samples(meta_valid, valid_probs_cal_4d, valid_targets, mapping)
    annual_valid = annual_valid[annual_valid["month_count"] == 12].copy()

    annual_calibration_method, _, annual_future_probs, annual_calibration_table = select_annual_calibration(
        annual_valid,
        annual_raw_future,
    )
    annual_calibration_table.to_csv(annual_calibration_csv, index=False, encoding="utf-8-sig")
    annual_raw_future["annual_calibrated_probability"] = annual_future_probs
    annual_raw_future.to_csv(BASE / "annual_prediction_calibrated_2025_2027.csv", index=False, encoding="utf-8-sig")

    final_metrics_row = {
        "selected_threshold": selected_threshold,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "pr_auc": calibrated_test_prob_metrics["pr_auc"],
        "roc_auc": calibrated_test_prob_metrics["roc_auc"],
        "brier": calibrated_test_prob_metrics["brier"],
        "raw_brier_before_calibration": raw_test_prob_metrics["brier"],
        "accuracy": test_metrics["accuracy"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
        "prevalence": calibrated_test_prob_metrics["prevalence"],
        "baseline_brier": calibrated_test_prob_metrics["baseline_brier"],
    }
    pd.DataFrame([final_metrics_row]).to_csv(test_metrics_csv, index=False, encoding="utf-8-sig")

    plot_pr_curve(test_targets_flat, clip_probs(test_probs_cal), pr_curve_png)
    plot_roc_curve(test_targets_flat, clip_probs(test_probs_cal), roc_curve_png)
    plot_calibration_curve(calibration_table, calibration_curve_png)
    plot_threshold_metrics(validation_threshold_table, selected_threshold, threshold_metrics_png)

    train_end = pd.Period(meta_train["target_start_month"].max(), freq="M")
    valid_start = pd.Period(meta_valid["target_start_month"].min(), freq="M")
    valid_end = pd.Period(meta_valid["target_start_month"].max(), freq="M")
    test_start = pd.Period(meta_test["target_start_month"].min(), freq="M")
    chronology_checks = {
        "train_before_valid": bool(train_end < valid_start),
        "valid_before_test": bool(valid_end < test_start),
    }

    interpretation_text = "\n".join(
        [
            f"validation set에서 recall {selection_recall_floor:.2f} 이상 후보 중 threshold {selected_threshold:.2f}를 선택했습니다." if not used_fallback else f"recall floor를 만족하는 threshold가 없어 F1 최대 threshold {selected_threshold:.2f}를 선택했습니다.",
            f"월별 calibration은 {monthly_calibration_method}, 연간 calibration은 {annual_calibration_method}가 선택되었습니다.",
            f"test PR-AUC={calibrated_test_prob_metrics['pr_auc']:.4f}, Brier={calibrated_test_prob_metrics['brier']:.4f}, baseline Brier={calibrated_test_prob_metrics['baseline_brier']:.4f} 입니다.",
            "월별 공식 평가는 next-month one-step 예측 기준이며, 연간 지도는 월별 calibrated 확률을 12개월 집계해 생성했습니다.",
        ]
    )

    eval_summary = {
        "pred_years": sorted(annual_raw_future["pred_year"].astype(int).unique().tolist()),
        "temperature": float(temperature_value),
        "selected_threshold": selected_threshold,
        "threshold_selection_method": "recall>=0.70 then max_f1 tie_precision else global max_f1",
        "threshold_selection_rule": "validation only, thresholds 0.10 to 0.90 step 0.01",
        "threshold_selection_recall_floor": float(selection_recall_floor),
        "threshold_selection_used_fallback": bool(used_fallback),
        "monthly_calibration_method": monthly_calibration_method,
        "annual_calibration_method": annual_calibration_method,
        "time_split_check": chronology_checks,
        "valid_month_min": str(meta_valid["target_start_month"].min()),
        "valid_month_max": str(meta_valid["target_start_month"].max()),
        "test_month_min": str(meta_test["target_start_month"].min()),
        "test_month_max": str(meta_test["target_start_month"].max()),
        "monthly_valid_best_metrics": best_valid,
        "monthly_test_metrics": test_metrics,
        "monthly_raw_test_prob_metrics": raw_test_prob_metrics,
        "monthly_test_prob_metrics": calibrated_test_prob_metrics,
        "pr_auc": calibrated_test_prob_metrics["pr_auc"],
        "roc_auc": calibrated_test_prob_metrics["roc_auc"],
        "brier": calibrated_test_prob_metrics["brier"],
        "baseline_brier": calibrated_test_prob_metrics["baseline_brier"],
        "prevalence": calibrated_test_prob_metrics["prevalence"],
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "validation_threshold_table_csv": str(validation_threshold_csv),
        "test_final_metrics_csv": str(test_metrics_csv),
        "monthly_calibration_comparison_csv": str(monthly_calibration_csv),
        "annual_calibration_comparison_csv": str(annual_calibration_csv),
        "calibration_curve_csv": str(calibration_table_csv),
        "pr_curve_png": str(pr_curve_png),
        "roc_curve_png": str(roc_curve_png),
        "calibration_curve_png": str(calibration_curve_png),
        "threshold_metrics_png": str(threshold_metrics_png),
        "interpretation": interpretation_text,
    }

    with open(eval_summary_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)

    print("=== monthly calibration 후보 비교 ===")
    print(monthly_calibration_table)
    print("\n=== annual calibration 후보 비교 ===")
    print(annual_calibration_table)
    print("\n=== validation threshold 탐색표 ===")
    print(validation_threshold_table)
    print("\n=== test 최종 성능표 ===")
    print(pd.DataFrame([final_metrics_row]))
    print("\n=== 자동 해석 ===")
    print(interpretation_text)
    print("\nsaved:", eval_summary_json)


if __name__ == "__main__":
    main()
