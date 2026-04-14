from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("/content") if Path("/content").exists() else SCRIPT_DIR

INPUT_WEEKLY_CSV_CANDIDATES = [
    BASE_DIR / "seoul_grid_week_full_panel_enriched_min.csv",
    BASE_DIR / "seoul_grid_week_full_panel_enriched.csv",
]
INPUT_EXTERNAL_CSV = BASE_DIR / "grid_external_features.csv"
INPUT_ADDITIONAL_MONTHLY_CSV = BASE_DIR / "grid_month_dynamic_features_additional.csv"
INPUT_ADDITIONAL_YEARLY_CSV = BASE_DIR / "grid_year_features_additional.csv"
GRID_GPKG = BASE_DIR / "seoul_grid_500m.gpkg"
EVENT_CSV = BASE_DIR / "seoul_pedestrian_vehicle_events_2007_2024_with_lonlat.csv"
FEATURE_SPEC_JSON = BASE_DIR / "feature_spec.json"
FEATURE_SPEC_EXTENSIONS_JSON = BASE_DIR / "feature_spec_extensions.json"

OUTPUT_MONTH_PANEL_CSV = BASE_DIR / "seoul_grid_month_panel_with_all_features_target.csv"
OUTPUT_MAPPING_CSV = BASE_DIR / "grid_id_to_row_col_mapping_rebuilt.csv"
OUTPUT_DIR = BASE_DIR / "convlstm_monthly_tensors"
OUTPUT_FEATURE_SPEC_JSON = OUTPUT_DIR / "feature_spec_resolved.json"
OUTPUT_FEATURE_DIAGNOSTICS_CSV = OUTPUT_DIR / "feature_diagnostics.csv"
OUTPUT_BUILD_SUMMARY_JSON = OUTPUT_DIR / "build_summary.json"
OUTPUT_NORMALIZATION_JSON = OUTPUT_DIR / "feature_normalization_stats.json"

SEQUENCE_LENGTH = 24
PREDICTION_HORIZON = 1
TARGET_COL = "target_next_month"
SUPERVISION_COL = "accident_occurred"
TENSOR_DTYPE = np.float16
ACCIDENT_PROXY_THRESHOLD = 0.5
MONTHS_SINCE_CAP = SEQUENCE_LENGTH
FEATURE_SCALE_DIVISORS = {
    "생활인구수_주중": 1000.0,
    "생활인구수_주말": 1000.0,
    "학교거리": 1000.0,
    "차량교통량": 1000.0,
    "가로등밀도": 1000.0,
    "경제활동인구": 1000.0,
    "비경제활동인구": 1000.0,
}
ABLATED_GLOBAL_DYNAMIC_FEATURES = set()
NON_NORMALIZED_FEATURES = {
    "monthly_accident_count",
    "lag_1m",
    "lag_2m",
    "lag_3m",
    "lag_6m",
    "lag_12m",
    "recent_3m_sum",
    "recent_6m_sum",
    "recent_12m_sum",
    "months_since_last_accident",
    "보호구역여부",
    "학교반경내여부",
    "어린이보호구역여부",
    "일방통행여부",
    "공원접근성",
}
LOG1P_FEATURES = {
    "생활인구수_주중",
    "생활인구수_주말",
    "횡단보도개수",
    "교차로개수",
    "버스정류장개수",
    "과속방지턱개수",
    "지하철출입구개수",
    "학교거리",
    "차량교통량",
    "버스서비스강도",
    "가로등밀도",
    "보행등밀도",
    "바닥형보행신호등설치밀도",
    "강수량",
    "경제활동인구",
    "비경제활동인구",
    "편의점밀도",
    "음식점밀도",
    "recent_3m_mean",
    "recent_6m_mean",
    "recent_12m_mean",
    "recent_24m_sum",
    "recent_24m_mean",
    "recent_6m_max",
    "recent_12m_max",
    "recent_6m_std",
    "recent_12m_std",
    "neighbor_recent_3m_mean",
    "neighbor_recent_6m_mean",
    "neighbor_recent_12m_mean",
    "neighbor_recent_24m_mean",
    "lag_1m_per_traffic",
    "recent_3m_per_traffic",
    "recent_12m_per_traffic",
    "recent_3m_per_weekday_pop",
    "recent_12m_per_weekday_pop",
}

DEFAULT_FEATURE_SPEC = [
    {"name": "monthly_accident_count", "source_col": "monthly_accident_count", "kind": "dynamic_monthly", "agg": "sum", "fill": "zero"},
    {"name": "lag_1m", "source_col": "monthly_accident_count", "kind": "derived_lag", "agg": "sum", "fill": "zero", "lag": 1},
    {"name": "lag_2m", "source_col": "monthly_accident_count", "kind": "derived_lag", "agg": "sum", "fill": "zero", "lag": 2},
    {"name": "lag_3m", "source_col": "monthly_accident_count", "kind": "derived_lag", "agg": "sum", "fill": "zero", "lag": 3},
    {"name": "lag_6m", "source_col": "monthly_accident_count", "kind": "derived_lag", "agg": "sum", "fill": "zero", "lag": 6},
    {"name": "lag_12m", "source_col": "monthly_accident_count", "kind": "derived_lag", "agg": "sum", "fill": "zero", "lag": 12},
    {"name": "recent_3m_sum", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "rolling_sum", "window": 3},
    {"name": "recent_6m_sum", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "rolling_sum", "window": 6},
    {"name": "recent_12m_sum", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "rolling_sum", "window": 12},
    {"name": "recent_3m_mean", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_mean", "window": 3},
    {"name": "recent_6m_mean", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_mean", "window": 6},
    {"name": "recent_12m_mean", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_mean", "window": 12},
    {"name": "recent_24m_sum", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "rolling_sum", "window": 24},
    {"name": "recent_24m_mean", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_mean", "window": 24},
    {"name": "recent_6m_max", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "max", "fill": "zero", "mode": "rolling_max", "window": 6},
    {"name": "recent_12m_max", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "max", "fill": "zero", "mode": "rolling_max", "window": 12},
    {"name": "recent_6m_std", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_std", "window": 6},
    {"name": "recent_12m_std", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "mean", "fill": "zero", "mode": "rolling_std", "window": 12},
    {"name": "recent_3m_minus_prev_3m", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "recent_vs_previous_sum_diff", "window": 3},
    {"name": "recent_6m_minus_prev_6m", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "recent_vs_previous_sum_diff", "window": 6},
    {"name": "months_since_last_accident", "source_col": "monthly_accident_count", "kind": "derived_history", "agg": "sum", "fill": "zero", "mode": "months_since_last_accident"},
    {"name": "생활인구수_주중", "source_col": "생활인구수_주중", "kind": "dynamic_monthly", "agg": "mean", "fill": "ffill"},
    {"name": "생활인구수_주말", "source_col": "생활인구수_주말", "kind": "dynamic_monthly", "agg": "mean", "fill": "ffill"},
    {"name": "횡단보도개수", "source_col": "횡단보도개수", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "교차로개수", "source_col": "교차로개수", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "버스정류장개수", "source_col": "버스정류장개수", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "보호구역여부", "source_col": "보호구역여부", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "과속방지턱개수", "source_col": "과속방지턱개수", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "지하철출입구개수", "source_col": "지하철출입구개수", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "용도혼합도", "source_col": "용도혼합도", "kind": "yearly_expand", "agg": "mean", "fill": "ffill"},
    {"name": "상업비율", "source_col": "상업비율", "kind": "yearly_expand", "agg": "mean", "fill": "ffill"},
    {"name": "주거비율", "source_col": "주거비율", "kind": "yearly_expand", "agg": "mean", "fill": "ffill"},
    {"name": "학교거리", "source_col": "학교거리", "kind": "static_grid", "agg": "mean", "fill": "static_ffill"},
    {"name": "학교반경내여부", "source_col": "학교반경내여부", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "어린이보호구역여부", "source_col": "어린이보호구역여부", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "일방통행여부", "source_col": "일방통행여부", "kind": "static_grid", "agg": "max", "fill": "static_ffill"},
    {"name": "month_sin", "source_col": "month", "kind": "derived_calendar", "agg": "mean", "fill": "ffill", "mode": "sin"},
    {"name": "month_cos", "source_col": "month", "kind": "derived_calendar", "agg": "mean", "fill": "ffill", "mode": "cos"},
    {"name": "neighbor_lag_1m_mean", "source_col": "lag_1m", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_lag_3m_mean", "source_col": "lag_3m", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_lag_12m_mean", "source_col": "lag_12m", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_recent_3m_mean", "source_col": "recent_3m_sum", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_recent_6m_mean", "source_col": "recent_6m_sum", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_recent_12m_mean", "source_col": "recent_12m_sum", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "neighbor_recent_24m_mean", "source_col": "recent_24m_sum", "kind": "derived_spatial", "agg": "mean", "fill": "zero", "mode": "neighbor_mean"},
    {"name": "lag_1m_per_traffic", "source_col": "lag_1m", "kind": "derived_exposure", "agg": "mean", "fill": "zero", "denominator_col": "차량교통량"},
    {"name": "recent_3m_per_traffic", "source_col": "recent_3m_sum", "kind": "derived_exposure", "agg": "mean", "fill": "zero", "denominator_col": "차량교통량"},
    {"name": "recent_12m_per_traffic", "source_col": "recent_12m_sum", "kind": "derived_exposure", "agg": "mean", "fill": "zero", "denominator_col": "차량교통량"},
    {"name": "recent_3m_per_weekday_pop", "source_col": "recent_3m_sum", "kind": "derived_exposure", "agg": "mean", "fill": "zero", "denominator_col": "생활인구수_주중"},
    {"name": "recent_12m_per_weekday_pop", "source_col": "recent_12m_sum", "kind": "derived_exposure", "agg": "mean", "fill": "zero", "denominator_col": "생활인구수_주중"},
]

REQUIRED_SPEC_KEYS = {"name", "source_col", "kind", "agg", "fill"}
ALLOWED_KINDS = {
    "dynamic_monthly",
    "yearly_expand",
    "static_grid",
    "derived_lag",
    "derived_history",
    "derived_calendar",
    "derived_spatial",
    "derived_exposure",
}
ALLOWED_AGGS = {"sum", "mean", "max"}
ALLOWED_FILLS = {"zero", "ffill", "static_ffill"}

RAW_CONTEXT_DEFAULTS = {
    "노인보호구역여부": 0.0,
    "어린이보호구역여부": 0.0,
    "주거": 0.0,
    "상업": 0.0,
    "공업": 0.0,
    "녹지": 0.0,
    "합계": 0.0,
}


def load_feature_spec() -> list[dict]:
    feature_spec = list(DEFAULT_FEATURE_SPEC)
    if FEATURE_SPEC_JSON.exists():
        with FEATURE_SPEC_JSON.open("r", encoding="utf-8") as fp:
            feature_spec = json.load(fp)

    if FEATURE_SPEC_EXTENSIONS_JSON.exists():
        with FEATURE_SPEC_EXTENSIONS_JSON.open("r", encoding="utf-8") as fp:
            extensions = json.load(fp)
        if not isinstance(extensions, list):
            raise ValueError("feature_spec_extensions.json은 리스트여야 합니다.")
        feature_spec = list(feature_spec) + extensions

    if not isinstance(feature_spec, list) or not feature_spec:
        raise ValueError("FEATURE_SPEC는 비어 있지 않은 리스트여야 합니다.")

    normalized = []
    seen_names = set()
    for idx, item in enumerate(feature_spec):
        if not isinstance(item, dict):
            raise ValueError(f"FEATURE_SPEC[{idx}]는 dict여야 합니다.")
        missing = REQUIRED_SPEC_KEYS - set(item)
        if missing:
            raise ValueError(f"FEATURE_SPEC[{idx}] 필수 키 누락: {sorted(missing)}")
        kind = item["kind"]
        agg = item["agg"]
        fill = item["fill"]
        enabled = item.get("enabled", True)
        if not enabled:
            continue
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"지원하지 않는 kind: {kind}")
        if agg not in ALLOWED_AGGS:
            raise ValueError(f"지원하지 않는 agg: {agg}")
        if fill not in ALLOWED_FILLS:
            raise ValueError(f"지원하지 않는 fill: {fill}")
        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"FEATURE_SPEC에 중복 feature name이 있습니다: {name}")
        seen_names.add(name)
        normalized_item = dict(item)
        if kind == "derived_lag" and "lag" not in normalized_item:
            raise ValueError(f"derived_lag feature에는 lag 값이 필요합니다: {name}")
        if kind == "derived_history":
            mode = normalized_item.get("mode")
            if mode not in {
                "rolling_sum",
                "rolling_mean",
                "rolling_max",
                "rolling_std",
                "recent_vs_previous_sum_diff",
                "months_since_last_accident",
            }:
                raise ValueError(f"derived_history feature의 mode가 올바르지 않습니다: {name}")
            if mode in {"rolling_sum", "rolling_mean", "rolling_max", "rolling_std", "recent_vs_previous_sum_diff"} and "window" not in normalized_item:
                raise ValueError(f"rolling_sum derived_history feature에는 window 값이 필요합니다: {name}")
        if kind == "derived_calendar":
            mode = normalized_item.get("mode")
            if mode not in {"sin", "cos"}:
                raise ValueError(f"derived_calendar feature의 mode가 올바르지 않습니다: {name}")
        if kind == "derived_spatial":
            mode = normalized_item.get("mode")
            if mode not in {"neighbor_mean"}:
                raise ValueError(f"derived_spatial feature의 mode가 올바르지 않습니다: {name}")
        if kind == "derived_exposure" and "denominator_col" not in normalized_item:
            raise ValueError(f"derived_exposure feature에는 denominator_col 값이 필요합니다: {name}")
        normalized.append(normalized_item)

    return normalized


def read_csv_header(path: Path) -> list[str]:
    try:
        return pd.read_csv(path, nrows=0).columns.tolist()
    except pd.errors.ParserError:
        return pd.read_csv(path, nrows=0, engine="python", on_bad_lines="skip").columns.tolist()


def resolve_weekly_input_path() -> Path:
    for path in INPUT_WEEKLY_CSV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "주간 패널 입력 파일이 없습니다.\n"
        + "\n".join(str(path) for path in INPUT_WEEKLY_CSV_CANDIDATES)
    )


def collect_required_weekly_columns(feature_spec: list[dict], weekly_header: list[str]) -> list[str]:
    source_cols = [spec["source_col"] for spec in feature_spec if spec["kind"] != "derived_lag"]
    weekly_usecols = ["grid_id", "year_week", "year", "week", "accident_count"]
    weekly_usecols += [col for col in source_cols if col in weekly_header]
    weekly_usecols += [col for col in RAW_CONTEXT_DEFAULTS if col in weekly_header]
    return sorted(set(weekly_usecols))


def read_weekly_panel(path: Path, usecols: list[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except pd.errors.ParserError:
        print("기본 CSV 파싱에 실패해 불량 행을 건너뛰고 다시 읽습니다.")
        return pd.read_csv(path, usecols=usecols, engine="python", on_bad_lines="skip")


def read_external_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["grid_id"])
    df = pd.read_csv(path)
    if "grid_id" not in df.columns:
        return pd.DataFrame(columns=["grid_id"])
    df["grid_id"] = df["grid_id"].astype(str)
    return df.drop_duplicates(subset=["grid_id"]).copy()


def read_additional_monthly_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["grid_id", "year_month"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["grid_id", "year_month"])
    if "grid_id" not in df.columns or "year_month" not in df.columns:
        raise ValueError("추가 월별 변수 파일에는 grid_id, year_month 컬럼이 필요합니다.")

    df["grid_id"] = df["grid_id"].astype(str)
    df["year_month"] = pd.PeriodIndex(df["year_month"].astype(str), freq="M").astype(str)
    if df.duplicated(["grid_id", "year_month"]).any():
        numeric_cols = [c for c in df.columns if c not in {"grid_id", "year_month"}]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = (
            df.groupby(["grid_id", "year_month"], as_index=False)[numeric_cols]
            .mean()
            .sort_values(["grid_id", "year_month"])
            .reset_index(drop=True)
        )
    return df


def read_additional_yearly_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["grid_id", "year"])

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["grid_id", "year"])
    if "grid_id" not in df.columns or "year" not in df.columns:
        raise ValueError("추가 연도별 변수 파일에는 grid_id, year 컬럼이 필요합니다.")

    df["grid_id"] = df["grid_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df


def detect_event_columns(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    date_candidates = ["date", "발생일", "발생일시", "accident_date"]
    lon_candidates = ["longitude", "lon", "x"]
    lat_candidates = ["latitude", "lat", "y"]
    date_col = next((c for c in date_candidates if c in columns), None)
    lon_col = next((c for c in lon_candidates if c in columns), None)
    lat_col = next((c for c in lat_candidates if c in columns), None)
    return date_col, lon_col, lat_col


def parse_year_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["grid_id"] = df["grid_id"].astype(str)

    if "year_week" in df.columns:
        split = df["year_week"].astype(str).str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
        year = pd.to_numeric(split["year"], errors="coerce")
        week = pd.to_numeric(split["week"], errors="coerce")
    else:
        year = pd.to_numeric(df["year"], errors="coerce")
        week = pd.to_numeric(df["week"], errors="coerce")

    valid = year.between(1900, 2100) & week.between(1, 53)
    df = df.loc[valid].copy()
    df["year"] = year.loc[valid].astype(int)
    df["week"] = week.loc[valid].astype(int)
    df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2)

    iso_key = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2) + "-1"
    df["week_start_date"] = pd.to_datetime(iso_key, format="%G-%V-%u", errors="coerce")
    df = df[df["week_start_date"].notna()].copy()
    df["year_month"] = df["week_start_date"].dt.to_period("M").astype(str)
    df["month"] = df["week_start_date"].dt.month.astype(int)
    return df


def load_grid_ids() -> tuple[list[str], gpd.GeoDataFrame | None]:
    if GRID_GPKG.exists():
        grid = gpd.read_file(GRID_GPKG)
        if "grid_id" in grid.columns:
            grid["grid_id"] = grid["grid_id"].astype(str)
            return sorted(grid["grid_id"].dropna().unique().tolist()), grid
    return [], None


def build_monthly_accidents_from_events(grid: gpd.GeoDataFrame | None) -> pd.DataFrame | None:
    if grid is None or not EVENT_CSV.exists():
        return None

    event_columns = read_csv_header(EVENT_CSV)
    date_col, lon_col, lat_col = detect_event_columns(event_columns)
    if date_col is None or lon_col is None or lat_col is None:
        return None

    events = pd.read_csv(EVENT_CSV, usecols=[date_col, lon_col, lat_col], low_memory=False)
    events[date_col] = pd.to_datetime(events[date_col], errors="coerce")
    events[lon_col] = pd.to_numeric(events[lon_col], errors="coerce")
    events[lat_col] = pd.to_numeric(events[lat_col], errors="coerce")
    events = events.dropna(subset=[date_col, lon_col, lat_col]).copy()
    if events.empty:
        return None

    if grid.crs is None:
        raise ValueError("격자 CRS가 없습니다. event 기반 월 라벨을 만들 수 없습니다.")

    event_gdf = gpd.GeoDataFrame(
        events,
        geometry=gpd.points_from_xy(events[lon_col], events[lat_col]),
        crs="EPSG:4326",
    ).to_crs(grid.crs)

    joined = gpd.sjoin(
        event_gdf,
        grid[["grid_id", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.dropna(subset=["grid_id"]).copy()
    if joined.empty:
        return None

    joined["grid_id"] = joined["grid_id"].astype(str)
    joined["year_month"] = joined[date_col].dt.to_period("M").astype(str)

    monthly = (
        joined.groupby(["grid_id", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "monthly_accident_count"})
    )
    monthly["monthly_accident_count"] = monthly["monthly_accident_count"].astype(np.float32)
    return monthly


def build_monthly_accidents_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        weekly.groupby(["grid_id", "year_month"], as_index=False)
        .agg(monthly_accident_count=("accident_count", "sum"))
    )
    monthly["monthly_accident_count"] = pd.to_numeric(monthly["monthly_accident_count"], errors="coerce").fillna(0).astype(np.float32)
    return monthly


def build_mapping_from_grid_id(grid_id_series: pd.Series) -> pd.DataFrame:
    grid_ids = pd.Series(sorted(grid_id_series.astype(str).dropna().unique()), name="grid_id")
    parsed = grid_ids.str.extract(r"SEOUL_R(?P<row>\d+)_C(?P<col>\d+)")
    if parsed.isna().any().any():
        bad = grid_ids[parsed.isna().any(axis=1)].head(10).tolist()
        raise ValueError(f"grid_id row/col 파싱 실패: {bad}")

    mapping = pd.DataFrame({"grid_id": grid_ids})
    mapping["row_raw"] = pd.to_numeric(parsed["row"], errors="coerce").astype(int)
    mapping["col_raw"] = pd.to_numeric(parsed["col"], errors="coerce").astype(int)

    row_values = sorted(mapping["row_raw"].unique())
    col_values = sorted(mapping["col_raw"].unique())
    row_map = {old: new for new, old in enumerate(row_values)}
    col_map = {old: new for new, old in enumerate(col_values)}

    mapping["row_idx"] = mapping["row_raw"].map(row_map).astype(int)
    mapping["col_idx"] = mapping["col_raw"].map(col_map).astype(int)
    return mapping.sort_values(["row_idx", "col_idx"]).reset_index(drop=True)


def aggregate_weekly_features(weekly: pd.DataFrame, feature_spec: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weekly = weekly.copy()
    dynamic_specs = [spec for spec in feature_spec if spec["kind"] == "dynamic_monthly" and spec["name"] != "monthly_accident_count"]
    yearly_specs = [spec for spec in feature_spec if spec["kind"] == "yearly_expand"]

    support_cols = []
    for col in RAW_CONTEXT_DEFAULTS:
        if col in weekly.columns:
            support_cols.append(col)

    static_like_sources = []
    for spec in feature_spec:
        if spec["kind"] == "static_grid" and spec["source_col"] in weekly.columns:
            static_like_sources.append(spec["source_col"])

    dynamic_agg_map = {}
    for spec in dynamic_specs:
        source_col = spec["source_col"]
        if source_col in weekly.columns:
            dynamic_agg_map[spec["name"]] = (source_col, spec["agg"])

    yearly_agg_map = {}
    for spec in yearly_specs:
        source_col = spec["source_col"]
        if source_col in weekly.columns:
            yearly_agg_map[spec["name"]] = (source_col, spec["agg"])

    static_agg_map = {}
    for source_col in sorted(set(static_like_sources + support_cols)):
        static_agg_map[source_col] = (source_col, "max")

    numeric_candidates = set()
    numeric_candidates.update([spec["source_col"] for spec in dynamic_specs if spec["source_col"] in weekly.columns])
    numeric_candidates.update([spec["source_col"] for spec in yearly_specs if spec["source_col"] in weekly.columns])
    numeric_candidates.update(static_agg_map.keys())
    for col in numeric_candidates:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce")

    monthly_dynamic = pd.DataFrame(columns=["grid_id", "year_month"])
    if dynamic_agg_map:
        monthly_dynamic = weekly.groupby(["grid_id", "year_month"], as_index=False).agg(**dynamic_agg_map)

    yearly_features = pd.DataFrame(columns=["grid_id", "year"])
    if yearly_agg_map:
        yearly_features = weekly.groupby(["grid_id", "year"], as_index=False).agg(**yearly_agg_map)

    grid_static = pd.DataFrame(columns=["grid_id"])
    if static_agg_map:
        grid_static = weekly.groupby("grid_id", as_index=False).agg(**static_agg_map)

    return monthly_dynamic, yearly_features, grid_static


def merge_external_static_features(grid_static: pd.DataFrame, external: pd.DataFrame, feature_spec: list[dict]) -> pd.DataFrame:
    result = grid_static.copy()
    if result.empty:
        result = pd.DataFrame({"grid_id": external["grid_id"].drop_duplicates()}) if "grid_id" in external.columns else pd.DataFrame(columns=["grid_id"])

    if external.empty:
        return result

    source_cols = [spec["source_col"] for spec in feature_spec if spec["kind"] == "static_grid"]
    keep_cols = ["grid_id"] + [col for col in source_cols if col in external.columns]
    keep_cols += [col for col in RAW_CONTEXT_DEFAULTS if col in external.columns and col not in keep_cols]
    if len(keep_cols) > 1:
        ext_subset = external[keep_cols].drop_duplicates(subset=["grid_id"]).copy()
        result = result.merge(ext_subset, on="grid_id", how="outer", suffixes=("", "_ext"))
        for col in list(result.columns):
            if col.endswith("_ext"):
                base_col = col[:-4]
                if base_col in result.columns:
                    result[base_col] = result[base_col].combine_first(result[col])
                    result = result.drop(columns=[col])
                else:
                    result = result.rename(columns={col: base_col})
    return result


def build_full_month_panel(
    all_grid_ids: list[str],
    month_range: list[str],
    target_monthly: pd.DataFrame,
    monthly_dynamic: pd.DataFrame,
    yearly_features: pd.DataFrame,
    grid_static: pd.DataFrame,
) -> pd.DataFrame:
    full_index = pd.MultiIndex.from_product([all_grid_ids, month_range], names=["grid_id", "year_month"])
    full_df = full_index.to_frame(index=False)

    full_df = full_df.merge(target_monthly, on=["grid_id", "year_month"], how="left")
    if not monthly_dynamic.empty:
        full_df = full_df.merge(monthly_dynamic, on=["grid_id", "year_month"], how="left")

    ym = full_df["year_month"].str.extract(r"(?P<year>\d{4})-(?P<month>\d{2})")
    full_df["year"] = pd.to_numeric(ym["year"], errors="coerce").astype(int)
    full_df["month"] = pd.to_numeric(ym["month"], errors="coerce").astype(int)

    if not yearly_features.empty:
        full_df = full_df.merge(yearly_features, on=["grid_id", "year"], how="left")

    if not grid_static.empty:
        full_df = full_df.merge(grid_static, on="grid_id", how="left")

    return full_df


def merge_additional_monthly_features(
    full_df: pd.DataFrame,
    additional_monthly: pd.DataFrame,
    feature_spec: list[dict],
) -> pd.DataFrame:
    if additional_monthly.empty:
        return full_df

    dynamic_source_cols = {
        spec["source_col"]
        for spec in feature_spec
        if spec["kind"] == "dynamic_monthly" and spec["source_col"] != "monthly_accident_count"
    }
    keep_cols = ["grid_id", "year_month"] + [col for col in dynamic_source_cols if col in additional_monthly.columns]
    if len(keep_cols) <= 2:
        return full_df

    monthly_subset = additional_monthly[keep_cols].copy()
    for col in keep_cols[2:]:
        monthly_subset[col] = pd.to_numeric(monthly_subset[col], errors="coerce")

    result = full_df.merge(monthly_subset, on=["grid_id", "year_month"], how="left", suffixes=("", "_extra"))
    for col in keep_cols[2:]:
        extra_col = f"{col}_extra"
        if extra_col in result.columns:
            if col in result.columns:
                result[col] = result[col].combine_first(result[extra_col])
                result = result.drop(columns=[extra_col])
            else:
                result = result.rename(columns={extra_col: col})
    return result


def merge_additional_yearly_features(
    yearly_features: pd.DataFrame,
    additional_yearly: pd.DataFrame,
    feature_spec: list[dict],
) -> pd.DataFrame:
    if additional_yearly.empty:
        return yearly_features

    result = yearly_features.copy()
    if result.empty:
        result = additional_yearly[["grid_id", "year"]].drop_duplicates().copy()

    yearly_source_cols = {
        spec["source_col"]
        for spec in feature_spec
        if spec["kind"] == "yearly_expand"
    }
    keep_cols = ["grid_id", "year"] + [col for col in yearly_source_cols if col in additional_yearly.columns]
    if len(keep_cols) <= 2:
        return result

    yearly_subset = additional_yearly[keep_cols].copy()
    for col in keep_cols[2:]:
        yearly_subset[col] = pd.to_numeric(yearly_subset[col], errors="coerce")

    result = result.merge(yearly_subset, on=["grid_id", "year"], how="outer", suffixes=("", "_extra"))
    for col in keep_cols[2:]:
        extra_col = f"{col}_extra"
        if extra_col in result.columns:
            if col in result.columns:
                result[col] = result[col].combine_first(result[extra_col])
                result = result.drop(columns=[extra_col])
            else:
                result = result.rename(columns={extra_col: col})
    return result


def fill_columns_by_spec(df: pd.DataFrame, feature_spec: list[dict]) -> pd.DataFrame:
    df = df.copy().sort_values(["grid_id", "year", "month"]).reset_index(drop=True)

    for spec in feature_spec:
        if spec["kind"] == "derived_lag":
            continue
        col = spec["name"]
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        if spec["fill"] == "zero":
            df[col] = series.fillna(0.0)
        elif spec["fill"] == "ffill":
            df[col] = (
                pd.DataFrame({"grid_id": df["grid_id"], "value": series})
                .groupby("grid_id")["value"]
                .transform(lambda s: s.ffill())
                .fillna(0.0)
            )
        elif spec["fill"] == "static_ffill":
            df[col] = (
                pd.DataFrame({"grid_id": df["grid_id"], "value": series})
                .groupby("grid_id")["value"]
                .transform(lambda s: s.ffill().bfill())
                .fillna(0.0)
            )

    df["monthly_accident_count"] = pd.to_numeric(df.get("monthly_accident_count", 0), errors="coerce").fillna(0.0).astype(np.float32)
    df["accident_occurred"] = (df["monthly_accident_count"] > 0).astype(np.float32)
    return df


def add_derived_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col, default in RAW_CONTEXT_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["보호구역여부"] = (
        ((df["노인보호구역여부"] > 0) | (df["어린이보호구역여부"] > 0)).astype(np.float32)
    )

    safe_total = df["합계"].where(df["합계"] > 0, np.nan)
    df["상업비율"] = (df["상업"] / safe_total).fillna(0.0).astype(np.float32)
    df["주거비율"] = (df["주거"] / safe_total).fillna(0.0).astype(np.float32)

    landuse_cols = ["주거", "상업", "공업", "녹지"]
    probs = np.vstack(
        [(df[col] / safe_total).fillna(0.0).to_numpy(dtype=np.float32) for col in landuse_cols]
    ).T

    entropy = np.zeros(len(df), dtype=np.float32)
    for idx in range(probs.shape[1]):
        p = probs[:, idx]
        mask = p > 0
        entropy[mask] += -(p[mask] * np.log(p[mask]))
    df["용도혼합도"] = entropy.astype(np.float32)
    return df


def add_targets_and_lags(df: pd.DataFrame, feature_spec: list[dict]) -> pd.DataFrame:
    df = df.copy().sort_values(["grid_id", "year", "month"]).reset_index(drop=True)
    lag_specs = [spec for spec in feature_spec if spec["kind"] == "derived_lag"]
    for spec in lag_specs:
        lag = int(spec["lag"])
        col = spec["name"]
        source_col = spec["source_col"]
        if source_col not in df.columns:
            df[col] = 0.0
            continue
        df[col] = df.groupby("grid_id")[source_col].shift(lag).fillna(0.0).astype(np.float32)

    def rolling_sum(series: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return values.rolling(window=window, min_periods=1).sum()

    def rolling_mean(series: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return values.rolling(window=window, min_periods=1).mean()

    def rolling_max(series: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return values.rolling(window=window, min_periods=1).max()

    def rolling_std(series: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return values.rolling(window=window, min_periods=2).std().fillna(0.0)

    def recent_vs_previous_sum_diff(series: pd.Series, window: int) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        current_sum = values.rolling(window=window, min_periods=1).sum()
        previous_sum = values.shift(window).rolling(window=window, min_periods=1).sum().fillna(0.0)
        return current_sum - previous_sum

    def months_since_last_accident(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        accident_flag = values >= ACCIDENT_PROXY_THRESHOLD
        out = np.empty(len(values), dtype=np.float32)
        last_seen = None
        for idx, flag in enumerate(accident_flag):
            if flag:
                last_seen = idx
                out[idx] = 0.0
            else:
                if last_seen is None:
                    out[idx] = float(MONTHS_SINCE_CAP)
                else:
                    out[idx] = float(min(idx - last_seen, MONTHS_SINCE_CAP))
        return pd.Series(out, index=series.index)

    history_specs = [spec for spec in feature_spec if spec["kind"] == "derived_history"]
    for spec in history_specs:
        col = spec["name"]
        source_col = spec["source_col"]
        if source_col not in df.columns:
            df[col] = 0.0
            continue

        mode = spec.get("mode")
        if mode == "rolling_sum":
            window = int(spec["window"])
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(lambda s, current_window=window: rolling_sum(s, current_window))
                .fillna(0.0)
                .astype(np.float32)
            )
        elif mode == "rolling_mean":
            window = int(spec["window"])
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(lambda s, current_window=window: rolling_mean(s, current_window))
                .fillna(0.0)
                .astype(np.float32)
            )
        elif mode == "rolling_max":
            window = int(spec["window"])
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(lambda s, current_window=window: rolling_max(s, current_window))
                .fillna(0.0)
                .astype(np.float32)
            )
        elif mode == "rolling_std":
            window = int(spec["window"])
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(lambda s, current_window=window: rolling_std(s, current_window))
                .fillna(0.0)
                .astype(np.float32)
            )
        elif mode == "recent_vs_previous_sum_diff":
            window = int(spec["window"])
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(lambda s, current_window=window: recent_vs_previous_sum_diff(s, current_window))
                .fillna(0.0)
                .astype(np.float32)
            )
        elif mode == "months_since_last_accident":
            df[col] = (
                df.groupby("grid_id")[source_col]
                .transform(months_since_last_accident)
                .fillna(float(MONTHS_SINCE_CAP))
                .astype(np.float32)
            )

    next_count = df.groupby("grid_id")["monthly_accident_count"].shift(-1)
    df["target_next_month_count"] = next_count
    df[TARGET_COL] = np.where(next_count.notna(), (next_count >= 1).astype(np.float32), np.nan)
    return df


def add_calendar_features(df: pd.DataFrame, feature_spec: list[dict]) -> pd.DataFrame:
    df = df.copy()
    calendar_specs = [spec for spec in feature_spec if spec["kind"] == "derived_calendar"]
    if not calendar_specs:
        return df

    month_num = pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int)
    angle = 2.0 * np.pi * (month_num - 1) / 12.0
    derived = {
        "sin": np.sin(angle).astype(np.float32),
        "cos": np.cos(angle).astype(np.float32),
    }

    for spec in calendar_specs:
        mode = spec.get("mode")
        if mode in derived:
            df[spec["name"]] = derived[mode]
    return df


def neighbor_mean_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    total = np.zeros_like(arr, dtype=np.float32)
    count = np.zeros_like(arr, dtype=np.float32)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = padded[1 + dr : 1 + dr + arr.shape[0], 1 + dc : 1 + dc + arr.shape[1]]
            total += shifted
            count += 1.0

    return np.divide(total, np.maximum(count, 1.0), dtype=np.float32)


def add_spatial_history_features(
    df: pd.DataFrame,
    feature_spec: list[dict],
    height: int,
    width: int,
) -> pd.DataFrame:
    df = df.copy()
    spatial_specs = [spec for spec in feature_spec if spec["kind"] == "derived_spatial"]
    if not spatial_specs:
        return df

    for spec in spatial_specs:
        if spec["name"] not in df.columns:
            df[spec["name"]] = 0.0

    month_groups = []
    for year_month, sub in df.groupby("year_month", sort=True):
        sub = sub.copy()
        row_idx = sub["row_idx"].to_numpy(dtype=np.int64)
        col_idx = sub["col_idx"].to_numpy(dtype=np.int64)

        for spec in spatial_specs:
            source_col = spec["source_col"]
            if source_col not in sub.columns:
                sub[spec["name"]] = 0.0
                continue

            grid = np.zeros((height, width), dtype=np.float32)
            values = pd.to_numeric(sub[source_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            grid[row_idx, col_idx] = values
            neigh = neighbor_mean_2d(grid)
            sub[spec["name"]] = neigh[row_idx, col_idx].astype(np.float32)

        month_groups.append(sub)

    return pd.concat(month_groups, ignore_index=True)


def add_exposure_features(df: pd.DataFrame, feature_spec: list[dict]) -> pd.DataFrame:
    df = df.copy()
    exposure_specs = [spec for spec in feature_spec if spec["kind"] == "derived_exposure"]
    if not exposure_specs:
        return df

    for spec in exposure_specs:
        numerator_col = spec["source_col"]
        denominator_col = spec["denominator_col"]
        output_col = spec["name"]

        if numerator_col not in df.columns or denominator_col not in df.columns:
            df[output_col] = 0.0
            continue

        numerator = pd.to_numeric(df[numerator_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        denominator = pd.to_numeric(df[denominator_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        safe_denominator = np.where(denominator > 0, denominator, np.nan)
        ratio = np.divide(numerator, safe_denominator, out=np.zeros_like(numerator, dtype=np.float32), where=np.isfinite(safe_denominator))
        df[output_col] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return df


def resolve_feature_spec(feature_spec: list[dict], available_columns: list[str]) -> list[dict]:
    resolved = []
    available_set = set(available_columns)
    for spec in feature_spec:
        if spec["name"] in ABLATED_GLOBAL_DYNAMIC_FEATURES:
            continue
        if spec["name"] in available_set:
            resolved.append(spec)
    return resolved


def month_weight(month_diff: int) -> float:
    if month_diff <= 23:
        return 1.0
    if month_diff <= 59:
        return 0.9
    if month_diff <= 119:
        return 0.8
    if month_diff <= 179:
        return 0.65
    return 0.5


def build_month_feature_map(
    sub: pd.DataFrame,
    feature_cols: list[str],
    height: int,
    width: int,
) -> np.ndarray:
    arr = np.zeros((len(feature_cols), height, width), dtype=TENSOR_DTYPE)
    row_idx = sub["row_idx"].to_numpy(dtype=np.int64)
    col_idx = sub["col_idx"].to_numpy(dtype=np.int64)

    for ch_idx, col in enumerate(feature_cols):
        values = pd.to_numeric(sub[col], errors="coerce").fillna(0).to_numpy(dtype=TENSOR_DTYPE)
        arr[ch_idx, row_idx, col_idx] = values
    return arr


def build_month_target_map(sub: pd.DataFrame, target_col: str, height: int, width: int) -> np.ndarray:
    arr = np.zeros((height, width), dtype=TENSOR_DTYPE)
    row_idx = sub["row_idx"].to_numpy(dtype=np.int64)
    col_idx = sub["col_idx"].to_numpy(dtype=np.int64)
    values = pd.to_numeric(sub[target_col], errors="coerce").fillna(0).to_numpy(dtype=TENSOR_DTYPE)
    arr[row_idx, col_idx] = values
    return arr


def split_indices(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = meta.copy()
    if "target_year" not in meta.columns or meta["target_year"].isna().all():
        fallback_col = "target_start_month" if "target_start_month" in meta.columns else "target_month"
        meta["target_year"] = pd.to_datetime(meta[fallback_col], errors="coerce").dt.year
    meta["target_year"] = pd.to_numeric(meta["target_year"], errors="coerce")

    train_mask = meta["target_year"] <= 2020
    valid_mask = meta["target_year"].between(2021, 2022)
    test_mask = meta["target_year"] >= 2023

    if train_mask.any() and valid_mask.any() and test_mask.any():
        return train_mask.values, valid_mask.values, test_mask.values

    years = sorted(meta["target_year"].dropna().astype(int).unique().tolist())
    if len(years) >= 4:
        test_years = {years[-1]}
        valid_years = set(years[-3:-1]) if len(years) >= 5 else {years[-2]}
        train_years = set(years) - valid_years - test_years

        train_mask = meta["target_year"].isin(train_years)
        valid_mask = meta["target_year"].isin(valid_years)
        test_mask = meta["target_year"].isin(test_years)
        if train_mask.any() and valid_mask.any() and test_mask.any():
            return train_mask.values, valid_mask.values, test_mask.values

    n = len(meta)
    if n < 3:
        raise ValueError(f"월단위 샘플 수가 너무 적습니다: {n}, years={years}")

    train_end = max(1, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    valid_end = min(valid_end, n - 1)

    train_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    valid_mask[train_end:valid_end] = True
    test_mask[valid_end:] = True
    return train_mask, valid_mask, test_mask


def build_feature_diagnostics(df: pd.DataFrame, resolved_feature_spec: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in resolved_feature_spec:
        col = spec["name"]
        rows.append(
            {
                "name": col,
                "source_col": spec["source_col"],
                "kind": spec["kind"],
                "agg": spec["agg"],
                "fill": spec["fill"],
                "exists": col in df.columns,
                "dtype": str(df[col].dtype) if col in df.columns else None,
                "missing_ratio": float(df[col].isna().mean()) if col in df.columns else 1.0,
                "nonzero_ratio": float((pd.to_numeric(df[col], errors="coerce").fillna(0) != 0).mean()) if col in df.columns else 0.0,
            }
        )
    return pd.DataFrame(rows)


def apply_feature_scaling(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        divisor = FEATURE_SCALE_DIVISORS.get(col)
        if divisor is None or divisor == 0:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) / divisor
    return df


def collect_train_input_months(sequence_rows: list[dict], train_mask: np.ndarray) -> list[str]:
    train_months = set()
    for keep, row in zip(train_mask, sequence_rows):
        if not keep:
            continue
        train_months.update(row["input_months"])
    return sorted(train_months)


def compute_feature_normalization_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_input_months: list[str],
) -> dict[str, dict[str, float | str]]:
    if not train_input_months:
        return {}

    train_df = df[df["year_month"].isin(train_input_months)].copy()
    stats: dict[str, dict[str, float | str]] = {}

    for col in feature_cols:
        if col not in train_df.columns or col in NON_NORMALIZED_FEATURES:
            continue

        values = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        transform = "log1p" if col in LOG1P_FEATURES else "identity"

        if transform == "log1p":
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))

        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        stats[col] = {
            "transform": transform,
            "mean": mean,
            "std": std,
        }

    return stats


def apply_feature_normalization(
    df: pd.DataFrame,
    normalization_stats: dict[str, dict[str, float | str]],
) -> pd.DataFrame:
    df = df.copy()
    for col, config in normalization_stats.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if config.get("transform") == "log1p":
            values = np.log1p(np.clip(values, a_min=0.0, a_max=None))

        mean = float(config.get("mean", 0.0))
        std = float(config.get("std", 1.0))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        df[col] = ((values - mean) / std).astype(np.float32)
    return df


def save_split(name: str, mask: np.ndarray, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, meta: pd.DataFrame) -> None:
    split_dir = OUTPUT_DIR
    np.save(split_dir / f"X_{name}.npy", X[mask].astype(TENSOR_DTYPE))
    np.save(split_dir / f"y_{name}.npy", y[mask].astype(TENSOR_DTYPE))
    np.save(split_dir / f"sample_weight_{name}.npy", sample_weight[mask].astype(np.float32))
    meta.loc[mask].reset_index(drop=True).to_csv(split_dir / f"meta_{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_spec = load_feature_spec()
    input_weekly_csv = resolve_weekly_input_path()
    print("[0/6] FEATURE_SPEC 로드 완료")
    print("    feature count:", len(feature_spec))

    header_cols = read_csv_header(input_weekly_csv)
    weekly_usecols = collect_required_weekly_columns(feature_spec, header_cols)

    print(f"[1/6] 주단위 패널 읽기: {input_weekly_csv}")
    weekly = read_weekly_panel(input_weekly_csv, usecols=weekly_usecols)
    weekly = parse_year_week(weekly)
    weekly["accident_count"] = pd.to_numeric(weekly["accident_count"], errors="coerce").fillna(0)

    external = read_external_features(INPUT_EXTERNAL_CSV)
    additional_monthly = read_additional_monthly_features(INPUT_ADDITIONAL_MONTHLY_CSV)
    additional_yearly = read_additional_yearly_features(INPUT_ADDITIONAL_YEARLY_CSV)
    all_grid_ids, grid = load_grid_ids()
    if not all_grid_ids:
        all_grid_ids = sorted(weekly["grid_id"].astype(str).dropna().unique().tolist())

    monthly_from_events = build_monthly_accidents_from_events(grid)
    if monthly_from_events is not None:
        print("    원본 event + grid 기반 월 사고건수를 사용합니다.")
        target_monthly = monthly_from_events.copy()
    else:
        print("    원본 event 데이터가 없거나 불완전해 주단위 패널에서 월 사고건수를 재구성합니다.")
        target_monthly = build_monthly_accidents_from_weekly(weekly)

    print("[2/6] 월단위 설명변수 패널 생성")
    monthly_dynamic, yearly_features, grid_static = aggregate_weekly_features(weekly, feature_spec)
    grid_static = merge_external_static_features(grid_static, external, feature_spec)
    yearly_features = merge_additional_yearly_features(yearly_features, additional_yearly, feature_spec)

    month_range = pd.period_range(
        weekly["year_month"].min(),
        weekly["year_month"].max(),
        freq="M",
    ).astype(str).tolist()

    month_panel = build_full_month_panel(
        all_grid_ids=all_grid_ids,
        month_range=month_range,
        target_monthly=target_monthly,
        monthly_dynamic=monthly_dynamic,
        yearly_features=yearly_features,
        grid_static=grid_static,
    )
    month_panel = merge_additional_monthly_features(month_panel, additional_monthly, feature_spec)
    month_panel = fill_columns_by_spec(month_panel, feature_spec)

    print("[3/6] row/col 매핑 생성")
    mapping = build_mapping_from_grid_id(pd.Series(all_grid_ids))
    mapping.to_csv(OUTPUT_MAPPING_CSV, index=False, encoding="utf-8-sig")
    month_panel = month_panel.merge(mapping[["grid_id", "row_idx", "col_idx"]], on="grid_id", how="left")
    height = int(mapping["row_idx"].max()) + 1
    width = int(mapping["col_idx"].max()) + 1

    month_panel = add_derived_context_features(month_panel)
    month_panel = add_targets_and_lags(month_panel, feature_spec)
    month_panel = add_calendar_features(month_panel, feature_spec)
    month_panel = add_spatial_history_features(month_panel, feature_spec, height, width)
    month_panel = add_exposure_features(month_panel, feature_spec)

    resolved_feature_spec = resolve_feature_spec(feature_spec, month_panel.columns.tolist())
    feature_cols = [spec["name"] for spec in resolved_feature_spec]
    month_panel = apply_feature_scaling(month_panel, feature_cols)

    print("[4/6] 월 시퀀스 메타 생성")
    month_order = (
        month_panel[["year_month", "year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    max_target_period = pd.Period(month_order["year_month"].iloc[-1], freq="M")
    weights = []
    meta_rows = []
    sequence_rows = []

    month_list = month_order["year_month"].tolist()
    for target_idx in range(SEQUENCE_LENGTH, len(month_list) - PREDICTION_HORIZON + 1):
        input_months = month_list[target_idx - SEQUENCE_LENGTH : target_idx]
        target_months = month_list[target_idx : target_idx + PREDICTION_HORIZON]
        target_month = target_months[0]
        target_end_month = target_months[-1]

        target_sub = month_panel[month_panel["year_month"].isin(target_months)]
        if target_sub["accident_occurred"].isna().all():
            continue

        target_period = pd.Period(target_month, freq="M")
        target_end_period = pd.Period(target_end_month, freq="M")
        diff = int(max_target_period.ordinal - target_end_period.ordinal)
        sample_weight = month_weight(diff)

        weights.append(sample_weight)
        sequence_rows.append(
            {
                "input_months": input_months,
                "target_months": target_months,
            }
        )
        meta_rows.append(
            {
                "sample_index": len(meta_rows),
                "input_start_month": input_months[0],
                "input_end_month": input_months[-1],
                "target_start_month": target_month,
                "target_end_month": target_end_month,
                "target_year": target_period.year,
                "target_month_num": target_period.month,
                "sample_weight": sample_weight,
            }
        )

    if not meta_rows:
        raise ValueError("월 시퀀스 샘플 메타가 생성되지 않았습니다. 입력 데이터 기간과 타깃 생성을 확인하세요.")

    sample_weight = np.array(weights, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    print("    target_year 분포:")
    print(meta["target_year"].value_counts().sort_index().to_string())

    train_mask, valid_mask, test_mask = split_indices(meta)

    train_input_months = collect_train_input_months(sequence_rows, train_mask)
    normalization_stats = compute_feature_normalization_stats(month_panel, feature_cols, train_input_months)
    month_panel = apply_feature_normalization(month_panel, normalization_stats)
    month_panel.to_csv(OUTPUT_MONTH_PANEL_CSV, index=False, encoding="utf-8-sig")

    month_to_x = {}
    month_to_y = {}
    for year_month in month_order["year_month"].tolist():
        sub = month_panel[month_panel["year_month"] == year_month].copy()
        month_to_x[year_month] = build_month_feature_map(sub, feature_cols, height, width)
        month_to_y[year_month] = build_month_target_map(sub, SUPERVISION_COL, height, width)

    X_samples = []
    y_samples = []
    for row in sequence_rows:
        X_samples.append(np.stack([month_to_x[m] for m in row["input_months"]], axis=0))
        y_samples.append(
            np.stack(
                [np.nan_to_num(month_to_y[m], nan=0.0).astype(TENSOR_DTYPE) for m in row["target_months"]],
                axis=0,
            )
        )

    X = np.stack(X_samples).astype(TENSOR_DTYPE)
    y = np.stack(y_samples).astype(TENSOR_DTYPE)

    base_sample_weight = sample_weight.copy()
    target_positive_ratio = y.reshape(len(y), -1).mean(axis=1).astype(np.float32)
    sample_weight = base_sample_weight * (1.0 + 1.5 * np.sqrt(np.clip(target_positive_ratio, 0.0, 1.0)))
    sample_weight = np.clip(sample_weight, 0.5, 3.0).astype(np.float32)
    meta["base_sample_weight"] = base_sample_weight
    meta["target_positive_ratio"] = target_positive_ratio
    meta["sample_weight"] = sample_weight

    print("[5/6] 텐서/메타/진단 파일 저장")
    save_split("train", train_mask, X, y, sample_weight, meta)
    save_split("valid", valid_mask, X, y, sample_weight, meta)
    save_split("test", test_mask, X, y, sample_weight, meta)

    diagnostics = build_feature_diagnostics(month_panel, resolved_feature_spec)
    diagnostics.to_csv(OUTPUT_FEATURE_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
    with OUTPUT_FEATURE_SPEC_JSON.open("w", encoding="utf-8") as fp:
        json.dump(resolved_feature_spec, fp, ensure_ascii=False, indent=2)
    with OUTPUT_NORMALIZATION_JSON.open("w", encoding="utf-8") as fp:
        json.dump(normalization_stats, fp, ensure_ascii=False, indent=2)

    build_summary = {
        "input_weekly_csv": str(input_weekly_csv),
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "target_col": TARGET_COL,
        "supervision_col": SUPERVISION_COL,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "grid_count": int(len(all_grid_ids)),
        "month_count": int(month_panel["year_month"].nunique()),
        "month_min": str(month_panel["year_month"].min()),
        "month_max": str(month_panel["year_month"].max()),
        "train_samples": int(train_mask.sum()),
        "valid_samples": int(valid_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "event_based_target": monthly_from_events is not None,
        "tensor_storage_dtype": str(np.dtype(TENSOR_DTYPE)),
        "feature_scale_divisors": FEATURE_SCALE_DIVISORS,
        "ablated_global_dynamic_features": sorted(ABLATED_GLOBAL_DYNAMIC_FEATURES),
        "normalized_feature_count": int(len(normalization_stats)),
        "normalization_stats_file": str(OUTPUT_NORMALIZATION_JSON),
    }
    with OUTPUT_BUILD_SUMMARY_JSON.open("w", encoding="utf-8") as fp:
        json.dump(build_summary, fp, ensure_ascii=False, indent=2)

    print("[6/6] 완료")
    print("    month panel :", OUTPUT_MONTH_PANEL_CSV)
    print("    mapping     :", OUTPUT_MAPPING_CSV)
    print("    tensor dir  :", OUTPUT_DIR)
    print("    feature cols:", feature_cols)
    print(f"    X shape     : {X.shape}")
    print(f"    y shape     : {y.shape}")
    print(f"    train/valid/test: {train_mask.sum()} / {valid_mask.sum()} / {test_mask.sum()}")


if __name__ == "__main__":
    main()
