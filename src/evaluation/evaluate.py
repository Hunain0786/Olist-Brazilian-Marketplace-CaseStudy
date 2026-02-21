import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import kagglehub

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def _load_pickle(name: str):
    with open(os.path.join(ARTIFACTS_DIR, name), "rb") as f:
        return pickle.load(f)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

def load_and_prepare_data():
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    customers   = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
    items       = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    reviews     = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
    orders      = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    sellers     = pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv"))
    geolocation = pd.read_csv(os.path.join(path, "olist_geolocation_dataset.csv"))

    df = pd.merge(orders, reviews, on="order_id")
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "review_creation_date",
        "review_answer_timestamp",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    delivered = df[df["order_status"] == "delivered"].copy()
    delivered["actual_delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.days
    delivered["estimated_delivery_days"] = (
        delivered["order_estimated_delivery_date"] - delivered["order_purchase_timestamp"]
    ).dt.days
    delivered["delivery_delay_days"] = (
        delivered["actual_delivery_days"] - delivered["estimated_delivery_days"]
    )

    delivered = pd.merge(
        delivered,
        customers[["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
        on="customer_id", how="left",
    )
    order_sellers = items.groupby("order_id")["seller_id"].first().reset_index()
    order_sellers = pd.merge(
        order_sellers,
        sellers[["seller_id", "seller_state", "seller_zip_code_prefix", "seller_city"]],
        on="seller_id", how="left",
    )
    delivered = pd.merge(
        delivered,
        order_sellers[["order_id", "seller_state", "seller_zip_code_prefix", "seller_city"]],
        on="order_id", how="left",
    )
    delivered["order_size"] = delivered["order_id"].map(
        items.groupby("order_id")["order_item_id"].count()
    )

    zip_coords = (
        geolocation
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
    )
    delivered["customer_lat"] = delivered["customer_zip_code_prefix"].map(zip_coords["geolocation_lat"])
    delivered["customer_lng"] = delivered["customer_zip_code_prefix"].map(zip_coords["geolocation_lng"])
    delivered["seller_lat"]   = delivered["seller_zip_code_prefix"].map(zip_coords["geolocation_lat"])
    delivered["seller_lng"]   = delivered["seller_zip_code_prefix"].map(zip_coords["geolocation_lng"])
    delivered["customer_seller_distance_km"] = haversine_km(
        delivered["customer_lat"], delivered["customer_lng"],
        delivered["seller_lat"],   delivered["seller_lng"],
    )
    median_dist = delivered["customer_seller_distance_km"].median()
    delivered["customer_seller_distance_km"] = delivered["customer_seller_distance_km"].fillna(median_dist)

    features_at_order_time = [
        "order_purchase_timestamp",
        "customer_state", "customer_city", "customer_zip_code_prefix",
        "seller_state",   "seller_city",   "seller_zip_code_prefix",
        "order_size", "customer_seller_distance_km",
    ]
    model_df = delivered[features_at_order_time + ["delivery_delay_days"]].copy()

    ts = pd.to_datetime(model_df["order_purchase_timestamp"])
    model_df["purchase_month"]       = ts.dt.month
    model_df["purchase_day_of_week"] = ts.dt.dayofweek
    model_df["purchase_hour"]        = ts.dt.hour
    model_df["is_weekend"]           = (model_df["purchase_day_of_week"] >= 5).astype(int)
    model_df["purchase_quarter"]     = ts.dt.quarter
    model_df = model_df.drop("order_purchase_timestamp", axis=1)

    state_to_region = _load_pickle("lookup_tables.pkl")["state_to_region"]
    model_df["state_pair"]       = model_df["customer_state"] + "_TO_" + model_df["seller_state"]
    model_df["customer_region"]  = model_df["customer_state"].map(state_to_region)
    model_df["seller_region"]    = model_df["seller_state"].map(state_to_region)
    model_df["region_pair"]      = model_df["customer_region"] + "_TO_" + model_df["seller_region"]

    overall_mean_delay = delivered["delivery_delay_days"].mean()
    city_mean_delay    = delivered.groupby("customer_city")["delivery_delay_days"].mean().to_dict()
    zip_mean_delay     = delivered.groupby("customer_zip_code_prefix")["delivery_delay_days"].mean().to_dict()
    zip_order_count    = delivered.groupby("customer_zip_code_prefix").size().to_dict()
    seller_zip_mean    = delivered.groupby("seller_zip_code_prefix")["delivery_delay_days"].mean().to_dict()

    clean_df = model_df.copy()
    clean_df["customer_city_avg_delay"]    = clean_df["customer_city"].map(city_mean_delay).fillna(overall_mean_delay)
    clean_df["customer_zip_avg_delay"]     = clean_df["customer_zip_code_prefix"].map(zip_mean_delay).fillna(overall_mean_delay)
    clean_df["customer_zip_order_volume"]  = clean_df["customer_zip_code_prefix"].map(zip_order_count).fillna(0)
    clean_df["seller_zip_avg_delay"]       = clean_df["seller_zip_code_prefix"].map(seller_zip_mean).fillna(overall_mean_delay)

    label_encoders = _load_pickle("label_encoders.pkl")
    categorical_cols = [
        "customer_state", "seller_state", "state_pair",
        "customer_region", "seller_region", "region_pair",
    ]
    for col in categorical_cols:
        le = label_encoders[col]
        clean_df[col + "_encoded"] = clean_df[col].astype(str).apply(
            lambda v, _le=le: int(_le.transform([v])[0]) if v in _le.classes_ else 0
        )

    clean_df = clean_df.drop(
        columns=categorical_cols + [
            "customer_city", "seller_city",
            "customer_zip_code_prefix", "seller_zip_code_prefix",
        ]
    )
    clean_df = clean_df.dropna()

    return clean_df

def evaluate():
    print("Loading model artifacts …")
    model   = _load_pickle("rf_regressor.pkl")
    scaler  = _load_pickle("scaler.pkl")
    lookups = _load_pickle("lookup_tables.pkl")

    feature_cols   = lookups["feature_cols"]
    numerical_cols = lookups["numerical_cols"]

    print("Preparing evaluation data …")
    clean_df = load_and_prepare_data()

    X = clean_df[feature_cols]
    y = clean_df["delivery_delay_days"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  DELIVERY DELAY MODEL — EVALUATION REPORT")
    print("=" * 50)
    print(f"  Test samples          : {len(y_test):,}")
    print(f"  Mean Absolute Error   : {mae:.2f} days")
    print(f"  Root Mean Squared Err : {rmse:.2f} days")
    print(f"  R² Score              : {r2:.4f}")
    print("=" * 50)

    errors = np.abs(y_test.values - y_pred)
    bands = [
        ("≤ 1 day",  errors <= 1),
        ("≤ 3 days", errors <= 3),
        ("≤ 5 days", errors <= 5),
        ("≤ 7 days", errors <= 7),
        ("> 7 days", errors > 7),
    ]
    print("\n  Error-band accuracy:")
    for label, mask in bands:
        pct = mask.sum() / len(errors) * 100
        print(f"    {label:>10s} : {pct:5.1f}% of predictions")

    def _categorise(days):
        if days > 7:
            return "Severely Late"
        elif days > 3:
            return "Moderately Late"
        elif days > 0:
            return "Slightly Late"
        elif days > -3:
            return "On Time"
        else:
            return "Early"

    actual_cats = pd.Series(y_test.values).apply(_categorise)
    pred_cats   = pd.Series(y_pred).apply(_categorise)
    cat_accuracy = (actual_cats.values == pred_cats.values).mean() * 100

    print(f"\n  Category-match accuracy : {cat_accuracy:.1f}%")
    print("  (Severely Late / Moderately Late / Slightly Late / On Time / Early)\n")

if __name__ == "__main__":
    evaluate()
