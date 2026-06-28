"""
Generate Daily Test Data

Samples from the original Delivery_Logistics.csv to produce 3 daily CSV files
(~5 000 records each) with a 74:26 no-delay / delay ratio.
Files are saved as:
    data/raw/daily_delivery_logistics_1.csv
    data/raw/daily_delivery_logistics_2.csv
    data/raw/daily_delivery_logistics_3.csv

Usage:
    python -m src.generate_daily_test_data_11

---------------------------------------------------------------------------
DATA PROVENANCE
---------------------------------------------------------------------------
Source dataset : data/raw/Delivery_Logistics.csv
Origin         : Kaggle — "Delivery Logistics Dataset (India – Multi-Partner)"
                 https://www.kaggle.com/datasets/kundanbedmutha/delivery-logistics-dataset-india-multi-partner
                 25 000 historical delivery records, publicly available.
Data type      : Stratified resample of the Kaggle source dataset
Generation     : Python (Pandas) — stratified sampling preserving the
                 74/26 on-time/delayed ratio of the original dataset.
Delay ratio    : ~26% delayed / ~74% on-time (matched to source)
Licence        : Kaggle dataset licence (source); MIT for this script
---------------------------------------------------------------------------
"""

import pandas as pd
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Delivery_Logistics.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

N_FILES = 3
RECORDS_PER_FILE = 5000
DELAY_RATIO = 0.26  # 26 % delayed, 74 % not delayed
RANDOM_SEED_BASE = 100

# Post-delivery columns: not available at prediction time
POST_DELIVERY_COLS = ["delivery_rating", "delivery_status"]


def generate() -> list[str]:
    df = pd.read_csv(SRC_PATH)

    delayed = df[df["delayed"] == "yes"]
    not_delayed = df[df["delayed"] == "no"]

    n_delayed = int(round(RECORDS_PER_FILE * DELAY_RATIO))  # ~1300
    n_not_delayed = RECORDS_PER_FILE - n_delayed              # ~3700

    saved: list[str] = []
    for i in range(1, N_FILES + 1):
        seed = RANDOM_SEED_BASE + i
        sample_delayed = delayed.sample(n=n_delayed, replace=True, random_state=seed)
        sample_not_delayed = not_delayed.sample(n=n_not_delayed, replace=True, random_state=seed)

        daily = (
            pd.concat([sample_not_delayed, sample_delayed], ignore_index=True)
            .sample(frac=1, random_state=seed)  # shuffle
            .reset_index(drop=True)
        )

        # Null out post-delivery columns (not available at prediction time)
        for col in POST_DELIVERY_COLS:
            if col in daily.columns:
                daily[col] = pd.NA

        out_path = OUT_DIR / f"daily_delivery_logistics_{i}.csv"
        daily.to_csv(out_path, index=False)
        saved.append(str(out_path))

        pct = daily["delayed"].value_counts(normalize=True)
        print(
            f"✓ File {i}: {out_path.name}  —  {len(daily)} rows  "
            f"(no: {pct.get('no', 0):.1%}, yes: {pct.get('yes', 0):.1%})"
        )

    return saved


if __name__ == "__main__":
    generate()
