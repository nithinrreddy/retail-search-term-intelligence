"""Data quality checks and advertising metrics (CTR, CVR, CPC, ROAS)."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")


def load_raw_terms():
    return pd.read_csv(DATA_DIR / "search_terms.csv")


def run_quality_checks(df: pd.DataFrame) -> dict:
    findings = {}

    missing = df.isna().sum()
    findings["missing_values"] = missing[missing > 0].to_dict()

    findings["duplicate_rows"] = int(df.duplicated().sum())
    findings["zero_impression_rows"] = int((df["impressions"] == 0).sum())

    findings["impossible_clicks_gt_impressions"] = int((df["clicks"] > df["impressions"]).sum())
    findings["impossible_orders_gt_clicks"] = int((df["orders"] > df["clicks"]).sum())

    numeric_cols = ["impressions", "clicks", "spend", "orders", "attributed_sales"]
    findings["negative_values"] = {
        col: int((df[col] < 0).sum()) for col in numeric_cols if (df[col] < 0).any()
    }

    return findings


def print_quality_report(findings: dict):
    print("DATA QUALITY REPORT")

    if findings["missing_values"]:
        print("Missing values found:")
        for col, count in findings["missing_values"].items():
            print(f"   - {col}: {count} row(s)")
    else:
        print("No missing values.")

    print(f"Duplicate rows: {findings['duplicate_rows']}")
    print(f"Zero-impression rows: {findings['zero_impression_rows']}")
    print(f"Rows with clicks > impressions: {findings['impossible_clicks_gt_impressions']}")
    print(f"Rows with orders > clicks: {findings['impossible_orders_gt_clicks']}")

    if findings["negative_values"]:
        print("Negative values found:")
        for col, count in findings["negative_values"].items():
            print(f"   - {col}: {count} row(s)")
    else:
        print("No negative values.")


def clean_terms(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first").copy()
    df["attributed_sales"] = df["attributed_sales"].fillna(0)  # missing sales = no sales
    after = len(df)
    print(f"Dropped {before - after} duplicate row(s). {after} rows remain.")
    return df


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
    df["cvr"] = np.where(df["clicks"] > 0, df["orders"] / df["clicks"], 0)
    df["cpc"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0)
    df["roas"] = np.where(df["spend"] > 0, df["attributed_sales"] / df["spend"], 0)
    return df


def main():
    raw = load_raw_terms()
    findings = run_quality_checks(raw)
    print_quality_report(findings)

    print()
    cleaned = clean_terms(raw)
    with_metrics = add_metrics(cleaned)

    print()
    print(with_metrics[["search_term", "ctr", "cvr", "cpc", "roas"]].head())

    out_path = DATA_DIR / "search_terms_clean.csv"
    with_metrics.to_csv(out_path, index=False)
    print(f"\nSaved cleaned data with metrics to {out_path}")


if __name__ == "__main__":
    main()
