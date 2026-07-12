"""
Smoke tests for FeatureEngineering (prediction_pipeline/src/feature_engineering_4.py).

Verifies that each derived feature the ML pipeline depends on actually gets
created — using a small synthetic DataFrame, not the real 5 000-row dataset.
No model files are loaded and no ML inference runs; feature engineering here
is pure pandas/sklearn-preprocessing arithmetic. The RandomForest models
themselves — and all MCP tool tests — live in tests/test_mcp_server.py and
are not duplicated here.

Run from the project root:
    uv run pytest tests/test_feature_engineering.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PP = PROJECT_ROOT / "prediction_pipeline"
sys.path.insert(0, str(PP))

from src.feature_engineering_4 import FeatureEngineering


# ---------------------------------------------------------------------------
# Small synthetic sample — column names match what FeatureEngineering expects
# after data_processing_3.py's cleaning step (slack_time, expected_time_hrs_num).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_df():
    return pd.DataFrame({
        "delivery_id":          ["D1", "D2", "D3", "D4"],
        "slack_time":           [2.0, -3.5, 0.0, -1.2],
        "delayed":              ["no", "yes", "no", "yes"],
        "distance_km":          [50.0, 220.5, 10.0, 180.0],
        "package_weight_kg":    [5.0, 25.0, 1.0, 15.0],
        "delivery_cost":        [100.0, 500.0, 20.0, 300.0],
        "expected_time_hrs_num": [4.0, 8.0, 1.0, 6.0],
        "weather_condition":    ["clear", "stormy", "foggy", "rainy"],
        "delivery_mode":        ["standard", "same day", "express", "two day"],
        "vehicle_type":         ["bike", "truck", "van", "ev"],
        "delivery_partner":     ["ekart", "xpressbees", "ekart", "dhl"],
        "region":               ["north", "east", "north", "west"],
    })


@pytest.fixture(scope="module")
def interaction_df(base_df):
    df, _ = FeatureEngineering.create_interaction_features(base_df, display=False)
    return df


# ── delay_hours / delayed encoding ──────────────────────────────────────────

def test_delay_hours_created_and_follows_slack_time_rule(base_df):
    df = FeatureEngineering.create_delay_hours(base_df, display=False)
    assert "delay_hours" in df.columns
    expected = base_df["slack_time"].apply(lambda x: 0 if x >= 0 else abs(x))
    assert (df["delay_hours"] == expected).all()


def test_encode_delayed_column_to_binary(base_df):
    df = FeatureEngineering.encode_delayed_column(base_df, display=False)
    assert set(df["delayed"].unique()) <= {0, 1}


# ── interaction features ─────────────────────────────────────────────────────

def test_interaction_features_exist(interaction_df):
    for col in ["weight_x_distance", "km_per_expected_hr", "cost_per_km", "cost_per_kg"]:
        assert col in interaction_df.columns, f"Missing interaction feature: {col}"


def test_weight_x_distance_formula(base_df, interaction_df):
    expected = base_df["package_weight_kg"] * base_df["distance_km"]
    assert (interaction_df["weight_x_distance"] == expected).all()


# ── ordinal / risk features ─────────────────────────────────────────────────

def test_ordinal_features_exist(interaction_df):
    df, _ = FeatureEngineering.create_ordinal_features(interaction_df, display=False)
    for col in ["weather_severity", "mode_urgency", "schedule_risk",
                "vehicle_capacity", "vehicle_load_strain"]:
        assert col in df.columns, f"Missing ordinal feature: {col}"


def test_schedule_risk_equals_weather_times_mode(interaction_df):
    df, _ = FeatureEngineering.create_ordinal_features(interaction_df, display=False)
    assert (df["schedule_risk"] == df["weather_severity"] * df["mode_urgency"]).all()


# ── group-aggregate features ─────────────────────────────────────────────────

def test_group_aggregate_features_exist(interaction_df):
    df, _ = FeatureEngineering.create_group_aggregate_features(interaction_df, display=False)
    for col in ["carrier_avg_schedule", "carrier_avg_weight", "region_avg_distance"]:
        assert col in df.columns, f"Missing group-aggregate feature: {col}"


# ── full pre-split pipeline ──────────────────────────────────────────────────

def test_pre_split_pipeline_creates_full_feature_set(base_df):
    df = FeatureEngineering.feature_eng_pre_split_pipeline(base_df, display=False)
    expected_features = {
        "delay_hours",
        "weight_x_distance", "km_per_expected_hr", "cost_per_km", "cost_per_kg",
        "weather_severity", "mode_urgency", "schedule_risk",
        "vehicle_capacity", "vehicle_load_strain",
        "carrier_avg_schedule", "carrier_avg_weight", "region_avg_distance",
    }
    missing = expected_features - set(df.columns)
    assert not missing, f"Pre-split pipeline did not create: {missing}"


# ── encoding / scaling (post-split) ──────────────────────────────────────────

def test_one_hot_encode_removes_categoricals(interaction_df):
    train, test = interaction_df.iloc[:2].copy(), interaction_df.iloc[2:].copy()
    train_enc, test_enc, encoded_cols = FeatureEngineering.one_hot_encode(
        train, test, categorical_columns=["weather_condition"], display=False,
    )
    assert "weather_condition" not in train_enc.columns
    assert len(encoded_cols) > 0


def test_scale_features_returns_fitted_scaler(interaction_df):
    train, test = interaction_df.iloc[:2].copy(), interaction_df.iloc[2:].copy()
    numeric_cols = ["distance_km", "package_weight_kg"]
    _, _, scaler = FeatureEngineering.scale_features(
        train, test, numeric_columns=numeric_cols, display=False,
    )
    assert scaler is not None
    assert len(scaler.mean_) == len(numeric_cols)
