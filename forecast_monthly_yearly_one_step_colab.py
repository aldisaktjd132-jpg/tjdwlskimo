from pathlib import Path
import json
import runpy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


BASE = Path("/content")
TENSOR_DIR = BASE / "convlstm_monthly_tensors"
MODEL_DIR = BASE / "convlstm_monthly_model"


def main() -> None:
    required = [
        BASE / "build_convlstm_monthly_tensors_from_grid_panel.py",
        BASE / "train_convlstm_monthly_colab_gpu.py",
        BASE / "seoul_grid_500m.gpkg",
        BASE / "seoul_pedestrian_vehicle_events_2007_2024_with_lonlat.csv",
        BASE / "seoul_grid_week_full_panel_enriched_min.csv",
        BASE / "grid_external_features.csv",
        BASE / "grid_month_dynamic_features_additional.csv",
        BASE / "grid_year_features_additional.csv",
        TENSOR_DIR / "feature_normalization_stats.json",
        MODEL_DIR / "best_convlstm_monthly.pt",
        TENSOR_DIR / "X_valid.npy",
        TENSOR_DIR / "y_valid.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 입력 파일이 없습니다.\n" + "\n".join(missing))

    build_module = runpy.run_path(str(BASE / "build_convlstm_monthly_tensors_from_grid_panel.py"))
    train_module = runpy.run_path(str(BASE / "train_convlstm_monthly_colab_gpu.py"))
    ConvLSTMForecaster = train_module["ConvLSTMForecaster"]

    load_feature_spec = build_module["load_feature_spec"]
    resolve_weekly_input_path = build_module["resolve_weekly_input_path"]
    read_csv_header = build_module["read_csv_header"]
    collect_required_weekly_columns = build_module["collect_required_weekly_columns"]
    read_weekly_panel = build_module["read_weekly_panel"]
    parse_year_week = build_module["parse_year_week"]
    read_external_features = build_module["read_external_features"]
    read_additional_monthly_features = build_module["read_additional_monthly_features"]
    read_additional_yearly_features = build_module["read_additional_yearly_features"]
    load_grid_ids = build_module["load_grid_ids"]
    build_monthly_accidents_from_events = build_module["build_monthly_accidents_from_events"]
    build_monthly_accidents_from_weekly = build_module["build_monthly_accidents_from_weekly"]
    aggregate_weekly_features = build_module["aggregate_weekly_features"]
    merge_external_static_features = build_module["merge_external_static_features"]
    merge_additional_yearly_features = build_module["merge_additional_yearly_features"]
    build_full_month_panel = build_module["build_full_month_panel"]
    merge_additional_monthly_features = build_module["merge_additional_monthly_features"]
    fill_columns_by_spec = build_module["fill_columns_by_spec"]
    build_mapping_from_grid_id = build_module["build_mapping_from_grid_id"]
    add_derived_context_features = build_module["add_derived_context_features"]
    add_targets_and_lags = build_module["add_targets_and_lags"]
    add_calendar_features = build_module["add_calendar_features"]
    add_spatial_history_features = build_module["add_spatial_history_features"]
    add_exposure_features = build_module["add_exposure_features"]
    resolve_feature_spec = build_module["resolve_feature_spec"]
    apply_feature_scaling = build_module["apply_feature_scaling"]
    apply_feature_normalization = build_module["apply_feature_normalization"]

    sequence_length = int(build_module["SEQUENCE_LENGTH"])

    feature_spec = load_feature_spec()
    input_weekly_csv = resolve_weekly_input_path()
    weekly_header = read_csv_header(input_weekly_csv)
    weekly_usecols = collect_required_weekly_columns(feature_spec, weekly_header)

    weekly = read_weekly_panel(input_weekly_csv, usecols=weekly_usecols)
    weekly = parse_year_week(weekly)
    weekly["accident_count"] = pd.to_numeric(weekly["accident_count"], errors="coerce").fillna(0)

    external = read_external_features(BASE / "grid_external_features.csv")
    additional_monthly = read_additional_monthly_features(BASE / "grid_month_dynamic_features_additional.csv")
    additional_yearly = read_additional_yearly_features(BASE / "grid_year_features_additional.csv")
    all_grid_ids, grid = load_grid_ids()
    if not all_grid_ids:
        all_grid_ids = sorted(weekly["grid_id"].astype(str).dropna().unique().tolist())

    monthly_from_events = build_monthly_accidents_from_events(grid)
    if monthly_from_events is not None:
        target_monthly = monthly_from_events.copy()
    else:
        target_monthly = build_monthly_accidents_from_weekly(weekly)

    monthly_dynamic, yearly_features, grid_static = aggregate_weekly_features(weekly, feature_spec)
    grid_static = merge_external_static_features(grid_static, external, feature_spec)
    yearly_features = merge_additional_yearly_features(yearly_features, additional_yearly, feature_spec)

    month_range = pd.period_range(
        weekly["year_month"].min(),
        weekly["year_month"].max(),
        freq="M",
    ).astype(str).tolist()

    month_panel_raw = build_full_month_panel(
        all_grid_ids=all_grid_ids,
        month_range=month_range,
        target_monthly=target_monthly,
        monthly_dynamic=monthly_dynamic,
        yearly_features=yearly_features,
        grid_static=grid_static,
    )
    month_panel_raw = merge_additional_monthly_features(month_panel_raw, additional_monthly, feature_spec)
    month_panel_raw = fill_columns_by_spec(month_panel_raw, feature_spec)

    mapping = build_mapping_from_grid_id(pd.Series(all_grid_ids))
    month_panel_raw = month_panel_raw.merge(mapping[["grid_id", "row_idx", "col_idx"]], on="grid_id", how="left")
    height = int(mapping["row_idx"].max()) + 1
    width = int(mapping["col_idx"].max()) + 1

    month_panel_raw = add_derived_context_features(month_panel_raw)
    month_panel_raw = add_targets_and_lags(month_panel_raw, feature_spec)
    month_panel_raw = add_calendar_features(month_panel_raw, feature_spec)
    month_panel_raw = add_spatial_history_features(month_panel_raw, feature_spec, height, width)
    month_panel_raw = add_exposure_features(month_panel_raw, feature_spec)

    resolved_feature_spec = resolve_feature_spec(feature_spec, month_panel_raw.columns.tolist())
    feature_cols = [spec["name"] for spec in resolved_feature_spec]

    with open(TENSOR_DIR / "feature_normalization_stats.json", "r", encoding="utf-8") as f:
        normalization_stats = json.load(f)

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

    X_valid = torch.tensor(np.load(TENSOR_DIR / "X_valid.npy"), dtype=torch.float32, device=device)
    y_valid = torch.tensor(np.load(TENSOR_DIR / "y_valid.npy"), dtype=torch.float32, device=device)

    with torch.no_grad():
        valid_logits = model(X_valid)

    temperature = torch.nn.Parameter(torch.tensor(1.0, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        t = torch.clamp(temperature, min=0.05, max=10.0)
        loss = F.binary_cross_entropy_with_logits(valid_logits / t, y_valid)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature_value = float(torch.clamp(temperature.detach(), min=0.05, max=10.0).cpu().item())

    base_exogenous_cols = [
        spec["name"]
        for spec in resolved_feature_spec
        if spec["kind"] in {"dynamic_monthly", "yearly_expand", "static_grid"} and spec["name"] != "monthly_accident_count"
    ]

    def build_model_ready_window(df_raw: pd.DataFrame, months: list[str]) -> pd.DataFrame:
        window_df = df_raw[df_raw["year_month"].isin(months)].copy()
        window_df = apply_feature_scaling(window_df, feature_cols)
        window_df = apply_feature_normalization(window_df, normalization_stats)
        return window_df

    def build_x_for_months(df_ready: pd.DataFrame, months: list[str]) -> np.ndarray:
        x_seq = []
        for ym in months:
            sub = df_ready[df_ready["year_month"] == ym].copy()
            feature_map = np.zeros((len(feature_cols), height, width), dtype=np.float32)
            rows = sub["row_idx"].astype(int).to_numpy()
            cols = sub["col_idx"].astype(int).to_numpy()
            for ch_idx, col in enumerate(feature_cols):
                feature_map[ch_idx, rows, cols] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            x_seq.append(feature_map)
        return np.stack(x_seq, axis=0)[None, ...]

    def build_future_exogenous_template(history_raw: pd.DataFrame, mapping_df: pd.DataFrame, target_period: pd.Period, exogenous_cols: list[str]) -> pd.DataFrame:
        same_month = history_raw[history_raw["month"] == target_period.month].copy()
        same_month = same_month.sort_values(["grid_id", "year", "month"])
        same_month_recent = same_month.groupby("grid_id", as_index=False).tail(3)
        if not same_month_recent.empty:
            same_month_recent = (
                same_month_recent.groupby("grid_id", as_index=False)[exogenous_cols]
                .median()
            )

        latest_any = history_raw.sort_values(["grid_id", "year", "month"]).groupby("grid_id", as_index=False).tail(1)
        recent_any = history_raw.sort_values(["grid_id", "year", "month"]).groupby("grid_id", as_index=False).tail(12)
        if not recent_any.empty:
            recent_any = recent_any.groupby("grid_id", as_index=False)[exogenous_cols].median()
        template = mapping_df[["grid_id", "row_idx", "col_idx"]].copy()

        if not same_month_recent.empty:
            template = template.merge(same_month_recent[["grid_id"] + exogenous_cols], on="grid_id", how="left")
        else:
            for col in exogenous_cols:
                template[col] = np.nan

        recent_any_subset = recent_any[["grid_id"] + exogenous_cols].copy() if not recent_any.empty else latest_any[["grid_id"] + exogenous_cols].copy()
        template = template.merge(recent_any_subset, on="grid_id", how="left", suffixes=("", "_fallback"))

        for col in exogenous_cols:
            fallback_col = f"{col}_fallback"
            if fallback_col in template.columns:
                template[col] = pd.to_numeric(template[col], errors="coerce").combine_first(
                    pd.to_numeric(template[fallback_col], errors="coerce")
                )
                template = template.drop(columns=[fallback_col])

        for col in exogenous_cols:
            template[col] = pd.to_numeric(template[col], errors="coerce").fillna(0.0)

        template["year_month"] = str(target_period)
        template["year"] = int(target_period.year)
        template["month"] = int(target_period.month)
        template["monthly_accident_count"] = 0.0
        template["accident_occurred"] = 0.0
        return template

    def refresh_recursive_features(df_raw: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.sort_values(["grid_id", "year", "month"]).reset_index(drop=True)
        df_raw = add_targets_and_lags(df_raw, feature_spec)
        df_raw = add_calendar_features(df_raw, feature_spec)
        df_raw = add_spatial_history_features(df_raw, feature_spec, height, width)
        df_raw = add_exposure_features(df_raw, feature_spec)
        return df_raw

    future_raw = month_panel_raw.copy()
    monthly_pred_rows = []

    for period in pd.period_range("2025-01", "2027-12", freq="M"):
        available_months = sorted(future_raw["year_month"].drop_duplicates().tolist())
        input_months = available_months[-sequence_length:]

        model_ready_window = build_model_ready_window(future_raw, input_months)
        X = build_x_for_months(model_ready_window, input_months)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(X_tensor) / temperature_value
            prob_map = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            raw_logit_map = logits.squeeze(0).squeeze(0).cpu().numpy()

        pred_df = mapping.copy()
        pred_df["pred_month"] = str(period)
        pred_df["pred_year"] = int(period.year)
        pred_df["pred_month_num"] = int(period.month)
        pred_df["raw_logit_next_month"] = raw_logit_map[pred_df["row_idx"], pred_df["col_idx"]]
        pred_df["calibrated_probability_next_month"] = prob_map[pred_df["row_idx"], pred_df["col_idx"]]
        monthly_pred_rows.append(pred_df.copy())

        next_month_df = build_future_exogenous_template(future_raw, mapping, period, base_exogenous_cols)
        next_month_df = next_month_df.sort_values("grid_id").reset_index(drop=True)
        next_month_df["monthly_accident_count"] = pred_df.sort_values("grid_id")["calibrated_probability_next_month"].to_numpy(dtype=np.float32)
        future_raw = pd.concat([future_raw, next_month_df], ignore_index=True, sort=False)
        future_raw = refresh_recursive_features(future_raw)

    monthly_result = pd.concat(monthly_pred_rows, ignore_index=True)
    annual_raw_result = (
        monthly_result
        .groupby(["pred_year", "grid_id", "row_idx", "col_idx"], as_index=False)
        .agg(
            annual_expected_monthly_positive_sum=("calibrated_probability_next_month", "sum"),
            annual_raw_probability=(
                "calibrated_probability_next_month",
                lambda s: float(1.0 - np.prod(1.0 - np.clip(s.to_numpy(dtype=float), 1e-6, 1 - 1e-6))),
            ),
        )
    )
    annual_raw_result["annual_risk_score"] = annual_raw_result["annual_expected_monthly_positive_sum"]

    monthly_result.to_csv(BASE / "monthly_prediction_probabilities_2025_2027.csv", index=False, encoding="utf-8-sig")
    annual_raw_result.to_csv(BASE / "annual_prediction_raw_2025_2027.csv", index=False, encoding="utf-8-sig")

    with open(MODEL_DIR / "temperature_scaling.json", "w", encoding="utf-8") as f:
        json.dump({"temperature": temperature_value}, f, ensure_ascii=False, indent=2)

    print("temperature:", temperature_value)
    print(monthly_result.head(5))
    print(annual_raw_result.head(5))


if __name__ == "__main__":
    main()
