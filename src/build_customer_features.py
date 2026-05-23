from pathlib import Path

import pandas as pd


def load_clean_data(processed_dir):
    processed_dir = Path(processed_dir)
    sales = pd.read_csv(processed_dir / "cleaned_sales_data.csv")
    customer = pd.read_csv(processed_dir / "cleaned_customer_info.csv")
    product = pd.read_csv(processed_dir / "cleaned_product_info.csv")
    return sales, customer, product


def build_merged_transactions(sales, customer, product):
    sales = sales.copy()
    customer = customer.copy()
    product = product.copy()

    sales["order_date"] = pd.to_datetime(sales["order_date"], errors="coerce")
    sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0)
    sales["unit_price"] = pd.to_numeric(sales["unit_price"], errors="coerce").fillna(0)

    if "discount_applied" in sales.columns:
        sales["discount_applied"] = pd.to_numeric(sales["discount_applied"], errors="coerce").fillna(0)
    else:
        sales["discount_applied"] = 0.0

    sales["line_amount"] = (sales["quantity"] * sales["unit_price"] * (1 - sales["discount_applied"])).round(2)

    merged = sales.merge(
        product[["product_id", "category"]],
        on="product_id",
        how="left",
    )
    merged = merged.merge(
        customer[["customer_id", "gender", "region", "loyalty_tier"]],
        on="customer_id",
        how="left",
        suffixes=("", "_customer"),
    )
    merged["category"] = merged["category"].fillna("Unknown")
    merged["gender"] = merged["gender"].fillna("Unknown")
    merged["region_customer"] = merged["region_customer"].fillna("Unknown")
    merged["loyalty_tier"] = merged["loyalty_tier"].fillna("Unknown")
    merged = merged.rename(columns={"region_customer": "customer_region"})

    return merged


def _preferred_category(merged):
    category_rank = (
        merged.groupby(["customer_id", "category"], as_index=False)["quantity"]
        .sum()
        .sort_values(["customer_id", "quantity", "category"], ascending=[True, False, True])
    )
    preferred = category_rank.drop_duplicates("customer_id", keep="first")
    return preferred[["customer_id", "category"]].rename(columns={"category": "preferred_category"})


def _purchase_frequency_from_dates(dates):
    unique_dates = pd.Series(dates).dropna().sort_values().drop_duplicates()
    if len(unique_dates) < 2:
        return 0.0
    day_gaps = unique_dates.diff().dropna().dt.days
    return float(day_gaps.mean())


def build_customer_features(merged, customer_info):
    merged = merged.copy()
    customer_info = customer_info.copy()

    reference_date = merged["order_date"].max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.today().normalize()

    customer_order_amount = (
        merged.groupby(["customer_id", "order_id"], as_index=False)["line_amount"]
        .sum()
        .rename(columns={"line_amount": "order_amount"})
    )

    order_agg = customer_order_amount.groupby("customer_id", as_index=False).agg(
        total_spent=("order_amount", "sum"),
        avg_order_value=("order_amount", "mean"),
        num_purchases=("order_id", "nunique"),
    )

    customer_agg = merged.groupby("customer_id", as_index=False).agg(
        total_quantity=("quantity", "sum"),
        first_purchase_date=("order_date", "min"),
        last_purchase_date=("order_date", "max"),
        has_discount=("discount_applied", lambda s: int((s > 0).any())),
    )

    customer_agg["days_since_first_purchase"] = (
        reference_date - customer_agg["first_purchase_date"]
    ).dt.days
    customer_agg["days_since_last_purchase"] = (
        reference_date - customer_agg["last_purchase_date"]
    ).dt.days

    purchase_frequency = (
        merged.groupby("customer_id")["order_date"]
        .apply(_purchase_frequency_from_dates)
        .reset_index(name="purchase_frequency")
    )
    preferred_category = _preferred_category(merged)

    customer_dim = customer_info[["customer_id", "gender", "region", "loyalty_tier"]].copy()
    customer_dim = customer_dim.rename(columns={"region": "customer_region"})
    customer_dim = customer_dim.drop_duplicates("customer_id")

    features = order_agg.merge(customer_agg, on="customer_id", how="outer")
    features = features.merge(purchase_frequency, on="customer_id", how="left")
    features = features.merge(preferred_category, on="customer_id", how="left")
    features = features.merge(customer_dim, on="customer_id", how="left")

    features["preferred_category"] = features["preferred_category"].fillna("Unknown")
    features["gender"] = features["gender"].fillna("Unknown")
    features["customer_region"] = features["customer_region"].fillna("Unknown")
    features["loyalty_tier"] = features["loyalty_tier"].fillna("Unknown")

    numeric_cols = [
        "total_spent",
        "avg_order_value",
        "total_quantity",
        "num_purchases",
        "days_since_first_purchase",
        "days_since_last_purchase",
        "purchase_frequency",
        "has_discount",
    ]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    features["total_spent"] = features["total_spent"].round(2)
    features["avg_order_value"] = features["avg_order_value"].round(2)
    features["purchase_frequency"] = features["purchase_frequency"].round(2)
    features["total_quantity"] = features["total_quantity"].astype(int)
    features["num_purchases"] = features["num_purchases"].astype(int)
    features["days_since_first_purchase"] = features["days_since_first_purchase"].astype(int)
    features["days_since_last_purchase"] = features["days_since_last_purchase"].astype(int)
    features["has_discount"] = features["has_discount"].astype(int)

    drop_cols = ["first_purchase_date", "last_purchase_date"]
    features = features.drop(columns=drop_cols, errors="ignore")
    features = features.sort_values("customer_id").reset_index(drop=True)

    return features


def save_outputs(merged, features, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_out = merged.copy()
    if "order_date" in merged_out.columns:
        merged_out["order_date"] = merged_out["order_date"].dt.strftime("%Y-%m-%d")

    merged_out.to_csv(output_dir / "merged_transactions.csv", index=False)
    features.to_csv(output_dir / "customer_features.csv", index=False)


def main():
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    sales, customer, product = load_clean_data(processed_dir)
    merged = build_merged_transactions(sales, customer, product)
    features = build_customer_features(merged, customer)
    save_outputs(merged, features, processed_dir)

    print("Saved merged and feature files to:", processed_dir)
    print("- merged_transactions.csv")
    print("- customer_features.csv")


if __name__ == "__main__":
    main()
