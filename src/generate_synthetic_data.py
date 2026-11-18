"""
Generate the synthetic datasets for Retail Search Term Intelligence.

Outputs (written to the data/ folder):
    products.csv            six fictional products
    search_terms.csv        one row per (product, search term) with 30-day ad metrics
    search_term_labels.csv  one relevance label per (product, search term)

All brands, products and numbers are made up. This data does not come from
Intentwise or any real advertiser. It exists to test the pipeline end to end.

Run from the project root:
    python src/generate_synthetic_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42  # Same seed gives the same data on every run, so results can be reproduced.
DATA_DIR = Path("data")
REPORT_DATE = "2026-10-31"  # End date of the 30-day reporting window.


# ---------------------------------------------------------------------------
# 1. Products
# ---------------------------------------------------------------------------
# Six fictional products across four categories.
PRODUCTS = [
    {"product_id": "P001", "product_title": "Stride Pro Running Shoe",
     "brand": "Northpeak", "category": "Running Shoes", "product_price": 89.99},
    {"product_id": "P002", "product_title": "TrailMax Waterproof Hiking Boot",
     "brand": "Ridgeline", "category": "Hiking Boots", "product_price": 129.00},
    {"product_id": "P003", "product_title": "CloudSoft Memory Foam Pillow",
     "brand": "Restwell", "category": "Bedding", "product_price": 39.99},
    {"product_id": "P004", "product_title": "ProBlend 900W Blender",
     "brand": "KitchenCore", "category": "Kitchen Appliances", "product_price": 59.99},
    {"product_id": "P005", "product_title": "Glow Vitamin C Serum 30ml",
     "brand": "Lumera", "category": "Skincare", "product_price": 24.50},
    {"product_id": "P006", "product_title": "Apexor Mesh Gaming Chair Pro",
     "brand": "Apexor", "category": "Gaming Chairs", "product_price": 199.00},
]


# ---------------------------------------------------------------------------
# 2. Manual annotations: search terms written by hand for each product
# ---------------------------------------------------------------------------
# The label is decided by a person reading the query in relation to the product.
# See docs/labelling_rules.md for the rules behind each group.
ANNOTATED_TERMS = {
    "P001": {  # Stride Pro Running Shoe
        "relevant": [
            "running shoes", "stride pro running shoe", "northpeak running shoes",
            "lightweight running shoes", "running shoes for daily training",
            "cushioned road running shoes", "northpeak stride pro", "men's road running shoes",
        ],
        "unrelated": [
            "hiking boots", "waterproof hiking boots", "yoga mat",
            "kitchen blender", "memory foam pillow", "tennis racket",
        ],
        "uncertain": [
            "shoes", "running gear", "athletic shoes sale", "nike running shoes",
        ],
    },
    "P002": {  # TrailMax Waterproof Hiking Boot
        "relevant": [
            "hiking boots", "waterproof hiking boots", "trailmax hiking boot",
            "ridgeline hiking boots", "waterproof trail boots", "hiking boots for men",
            "ankle support hiking boots", "trail hiking boots waterproof",
        ],
        "unrelated": [
            "running shoes", "stride pro", "camping tent",
            "snow shovel", "dress shoes", "dog leash",
        ],
        "uncertain": [
            "boots", "outdoor shoes", "timberland boots", "outdoor gear",
        ],
    },
    "P003": {  # CloudSoft Memory Foam Pillow
        "relevant": [
            "memory foam pillow", "cloudsoft pillow", "restwell memory foam pillow",
            "pillow for neck pain", "memory foam pillow queen", "contour memory foam pillow",
            "cooling memory foam pillow", "best pillow for side sleepers",
        ],
        "unrelated": [
            "down comforter", "bed sheets", "blender",
            "vitamin c serum", "mattress topper", "bath towels",
        ],
        "uncertain": [
            "pillow", "sleep products", "tempur pillow", "bedroom decor",
        ],
    },
    "P004": {  # ProBlend 900W Blender
        "relevant": [
            "blender", "900w blender", "problend blender", "kitchencore blender",
            "smoothie blender", "countertop blender 900 watt",
            "personal blender for smoothies", "kitchen blender with glass jar",
        ],
        "unrelated": [
            "coffee maker", "air fryer", "memory foam pillow",
            "running shoes", "blender bottle", "smoothie recipes",
        ],
        "uncertain": [
            "kitchen appliances", "vitamix blender", "ninja blender", "small kitchen gadgets",
        ],
    },
    "P005": {  # Glow Vitamin C Serum 30ml
        "relevant": [
            "vitamin c serum", "glow vitamin c serum", "lumera serum",
            "vitamin c serum for dark spots", "vitamin c face serum 30ml",
            "brightening vitamin c serum", "lumera vitamin c", "vitamin c serum for face",
        ],
        "unrelated": [
            "retinol cream", "sunscreen spf 50", "hair serum",
            "protein powder", "how to remove dark spots", "moisturizer for dry skin",
        ],
        "uncertain": [
            "skincare serum", "face serum", "tatcha vitamin c serum", "anti aging products",
        ],
    },
    "P006": {  # Apexor Mesh Gaming Chair Pro
        "relevant": [
            "gaming chair", "mesh gaming chair", "apexor gaming chair",
            "mesh gaming chair pro", "ergonomic gaming chair",
            "gaming chair with lumbar support", "office gaming chair mesh",
            "gaming chair under 200",
        ],
        "unrelated": [
            "office desk", "gaming mouse", "standing desk converter",
            "computer monitor", "how to assemble a chair", "gaming headset",
        ],
        "uncertain": [
            "chair", "office chair", "racing chair", "secretlab chair",
        ],
    },
}

# Short plain-language note saved with each label in search_term_labels.csv.
LABEL_NOTES = {
    "relevant": "Names this product type, brand or product name.",
    "unrelated": "Describes a different product type, or an informational intent.",
    "uncertain": "Too broad, or names another brand for the same product type.",
}


# ---------------------------------------------------------------------------
# 3. Advertising metrics
# ---------------------------------------------------------------------------
# Typical ranges for each label. These are assumptions for synthetic data only.
# Relevant terms are set to look healthier than unrelated ones, which matches
# the business intuition. Real advertisers' numbers will differ.
METRIC_RANGES = {
    "relevant":  {"impressions": (800, 4000), "ctr": (0.03, 0.08),
                  "cvr": (0.05, 0.15), "cpc": (0.60, 1.40)},
    "uncertain": {"impressions": (500, 2500), "ctr": (0.02, 0.05),
                  "cvr": (0.02, 0.08), "cpc": (0.70, 1.50)},
    "unrelated": {"impressions": (300, 2000), "ctr": (0.005, 0.02),
                  "cvr": (0.00, 0.02), "cpc": (0.80, 1.80)},
}


def make_metrics(label, price, rng):
    """Create one row of advertising metrics for a search term.

    Each number depends on the one before it:
    impressions -> clicks (using CTR) -> orders (using CVR) -> spend (using CPC).
    """
    ranges = METRIC_RANGES[label]
    impressions = int(rng.integers(*ranges["impressions"]))
    clicks = int(round(impressions * rng.uniform(*ranges["ctr"])))
    orders = int(rng.binomial(clicks, rng.uniform(*ranges["cvr"])))
    spend = round(clicks * rng.uniform(*ranges["cpc"]), 2)
    # Attributed sales are roughly orders times price, with a little noise.
    attributed_sales = round(orders * price * rng.uniform(0.9, 1.1), 2)
    return {
        "impressions": impressions,
        "clicks": clicks,
        "spend": spend,
        "orders": orders,
        "attributed_sales": attributed_sales,
    }


# ---------------------------------------------------------------------------
# 4. Build the tables
# ---------------------------------------------------------------------------
def build_tables(rng):
    """Turn the hand-written annotations into the three tables."""
    products_df = pd.DataFrame(PRODUCTS)

    term_rows = []
    label_rows = []
    for product in PRODUCTS:
        pid = product["product_id"]
        campaign = f"SP_{product['category'].replace(' ', '_')}_Core"

        for label, terms in ANNOTATED_TERMS[pid].items():
            for term in terms:
                metrics = make_metrics(label, product["product_price"], rng)
                term_rows.append({
                    "product_id": pid,
                    "campaign_name": campaign,
                    "search_term": term,
                    **metrics,
                    "report_date": REPORT_DATE,
                })
                label_rows.append({
                    "product_id": pid,
                    "search_term": term,
                    "relevance_label": label,
                    "annotation_notes": LABEL_NOTES[label],
                })

    return products_df, pd.DataFrame(term_rows), pd.DataFrame(label_rows)


def add_quality_issues(terms_df, rng):
    """Add realistic messiness so the data quality checks have something to find.

    Real advertising exports often contain missing values, zero-impression rows
    and duplicate lines. We add a few on purpose, and the README explains them.
    """
    df = terms_df.copy()

    # Issue 1: three missing sales values.
    missing_rows = rng.choice(len(df), size=3, replace=False)
    df.loc[missing_rows, "attributed_sales"] = np.nan

    # Issue 2: one row with zero impressions. CTR would divide by zero here.
    zero_row = int(rng.integers(len(df)))
    df.loc[zero_row, ["impressions", "clicks", "orders", "spend", "attributed_sales"]] = 0

    # Issue 3: one duplicated row.
    duplicate = df.iloc[[int(rng.integers(len(df)))]]
    df = pd.concat([df, duplicate], ignore_index=True)

    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    DATA_DIR.mkdir(exist_ok=True)

    products_df, terms_df, labels_df = build_tables(rng)

    # Shuffle so the rows are not grouped by product or label.
    terms_df = terms_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    terms_df = add_quality_issues(terms_df, rng)

    products_df.to_csv(DATA_DIR / "products.csv", index=False)
    terms_df.to_csv(DATA_DIR / "search_terms.csv", index=False)
    labels_df.to_csv(DATA_DIR / "search_term_labels.csv", index=False)

    print(f"products.csv:           {len(products_df)} rows")
    print(f"search_terms.csv:       {len(terms_df)} rows (includes intentional quality issues)")
    print(f"search_term_labels.csv: {len(labels_df)} rows")
    print("\nLabel counts:")
    print(labels_df["relevance_label"].value_counts())


if __name__ == "__main__":
    main()
