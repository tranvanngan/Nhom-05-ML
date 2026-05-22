from pathlib import Path
import pandas as pd

REGION_MAP = {
    "north": "North", "nrth": "North",
    "south": "South", "east": "East", "west": "West", "central": "Central",
}

DELIVERY_STATUS_MAP = {
    "delivered": "Delivered", "delrd": "Delivered",
    "delayed": "Delayed", "delyd": "Delayed",
    "cancelled": "Cancelled",
}

PAYMENT_METHOD_MAP = {
    "credit card": "Credit Card",
    "bank transfer": "Bank Transfer",
    "bank transfr": "Bank Transfer",
    "paypal": "PayPal",
}

GENDER_MAP = {
    "male": "Male",
    "female": "Female", "femle": "Female",
    "other": "Other",
}

LOYALTY_TIER_MAP = {
    "gold": "Gold", "gld": "Gold",
    "silver": "Silver", "sllver": "Silver",
    "bronze": "Bronze", "brnze": "Bronze",
}

QUANTITY_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}


def _clean_text(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return pd.NA if text == "" else text


def _map_category(value, mapping, default="Unknown"):
    text = _clean_text(value)
    if pd.isna(text):
        return default
    key = str(text).strip().lower()
    return mapping.get(key, str(text).strip().title())


def _parse_date(series):
    parsed = pd.to_datetime(series, format="%d-%m-%y", errors="coerce")
    remaining = parsed.isna() & series.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(series[remaining], errors="coerce")
    return parsed


def _fill_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce")
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 0
    return numeric.fillna(median_value)


def _fill_date(series):
    parsed = _parse_date(series)
    mode_value = parsed.mode(dropna=True)
    fill_value = mode_value.iloc[0] if not mode_value.empty else pd.Timestamp("1900-01-01")
    return parsed.fillna(fill_value).dt.strftime("%Y-%m-%d")


def load_raw_data(raw_dir):
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / "sales_data.csv")
    customer = pd.read_csv(raw_dir / "customer_info.csv")
    product = pd.read_csv(raw_dir / "product_info.csv")
    return sales, customer, product


def clean_sales(df):
    df = df.copy()

    df["order_id"] = df["order_id"].map(_clean_text).fillna("UNKNOWN_ORDER")
    df["customer_id"] = df["customer_id"].map(_clean_text).fillna("UNKNOWN_CUSTOMER")
    df["product_id"] = df["product_id"].map(_clean_text).fillna("UNKNOWN_PRODUCT")

    quantity_text = df["quantity"].map(_clean_text)
    quantity_text = quantity_text.map(lambda x: QUANTITY_WORDS.get(str(x).lower(), x) if not pd.isna(x) else pd.NA)
    df["quantity"] = _fill_numeric(quantity_text).round().astype(int)

    df["unit_price"] = _fill_numeric(df["unit_price"]).round(2)
    df["discount_applied"] = _fill_numeric(df["discount_applied"]).clip(0, 1).round(2)
    df["order_date"] = _fill_date(df["order_date"])

    df["delivery_status"] = df["delivery_status"].map(lambda x: _map_category(x, DELIVERY_STATUS_MAP))
    df["payment_method"] = df["payment_method"].map(lambda x: _map_category(x, PAYMENT_METHOD_MAP))
    df["region"] = df["region"].map(lambda x: _map_category(x, REGION_MAP))

    return df


def clean_customer(df):
    df = df.copy()

    df["customer_id"] = df["customer_id"].map(_clean_text).fillna("UNKNOWN_CUSTOMER")
    df["email"] = df["email"].map(_clean_text).fillna("Unknown")
    df["signup_date"] = _fill_date(df["signup_date"])
    df["gender"] = df["gender"].map(lambda x: _map_category(x, GENDER_MAP))
    df["region"] = df["region"].map(lambda x: _map_category(x, REGION_MAP))
    df["loyalty_tier"] = df["loyalty_tier"].map(lambda x: _map_category(x, LOYALTY_TIER_MAP))

    return df


def clean_product(df):
    df = df.copy()

    df["product_id"] = df["product_id"].map(_clean_text).fillna("UNKNOWN_PRODUCT")
    df["product_name"] = df["product_name"].map(_clean_text).fillna("Unknown")
    df["category"] = df["category"].map(_clean_text).fillna("Unknown")
    df["launch_date"] = _fill_date(df["launch_date"])
    df["base_price"] = _fill_numeric(df["base_price"]).round(2)
    df["supplier_code"] = df["supplier_code"].map(_clean_text).fillna("Unknown")

    return df


def missing_summary(tables):
    rows = []
    for table_name, df in tables.items():
        for column in df.columns:
            rows.append({
                "table": table_name,
                "column": column,
                "missing_count": int(df[column].isna().sum()),
                "missing_rate": round(float(df[column].isna().mean()), 4),
            })
    return pd.DataFrame(rows)


def describe_numeric(tables):
    parts = []
    for table_name, df in tables.items():
        num = df.select_dtypes(include="number")
        if not num.empty:
            stat = num.describe().T
            stat.insert(0, "table", table_name)
            stat.insert(1, "column", stat.index)
            parts.append(stat.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def iqr_outlier_summary(tables):
    rows = []
    for table_name, df in tables.items():
        for column in df.select_dtypes(include="number").columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((df[column] < lower) | (df[column] > upper)).sum()) if iqr != 0 else 0
            rows.append({
                "table": table_name,
                "column": column,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "outlier_count": count,
            })
    return pd.DataFrame(rows)


def save_clean_data(sales, customer, product, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sales.to_csv(output_dir / "cleaned_sales_data.csv", index=False)
    customer.to_csv(output_dir / "cleaned_customer_info.csv", index=False)
    product.to_csv(output_dir / "cleaned_product_info.csv", index=False)


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"

    sales_raw, customer_raw, product_raw = load_raw_data(raw_dir)
    sales = clean_sales(sales_raw)
    customer = clean_customer(customer_raw)
    product = clean_product(product_raw)

    save_clean_data(sales, customer, product, output_dir)
    print("Saved cleaned CSV files to:", output_dir)


if __name__ == "__main__":
    main()
