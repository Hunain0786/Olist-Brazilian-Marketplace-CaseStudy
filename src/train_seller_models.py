import os
import pickle
import kagglehub
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    print("Downloading / loading dataset …")
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
    
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    delivered = orders[orders["order_status"] == "delivered"].copy()
    
    df = items.merge(delivered, on="order_id", how="inner")
    
    return df, reviews


def extract_features(df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    total_revenue = df.groupby("seller_id")["price"].sum().rename("total_revenue")
    total_orders = df.groupby("seller_id")["order_id"].nunique().rename("total_orders")

    order_revenue = df.groupby(["seller_id", "order_id"])["price"].sum().reset_index()
    avg_order_value = order_revenue.groupby("seller_id")["price"].mean().rename("avg_order_value")

    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400
    avg_delivery_time = df.groupby("seller_id")["delivery_time_days"].mean().rename("avg_delivery_time_days")

    df["is_late"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)
    late_rate = df.groupby("seller_id")["is_late"].mean().rename("late_delivery_rate")

    order_freight = df.groupby(["seller_id", "order_id"])["freight_value"].sum().reset_index()
    avg_freight_per_order = order_freight.groupby("seller_id")["freight_value"].mean().rename("avg_freight_per_order")

    df["processing_time_days"] = (
        df["order_delivered_carrier_date"] - df["order_approved_at"]
    ).dt.total_seconds() / 86400
    avg_processing_time = df.groupby("seller_id")["processing_time_days"].mean().rename("avg_processing_time_days")

    reviews_orders = reviews[["order_id", "review_score"]].merge(
        df[["order_id"]].drop_duplicates(), on="order_id", how="inner"
    )
    reviews_sellers = reviews_orders.merge(
        df[["order_id", "seller_id"]].drop_duplicates(),
        on="order_id", how="inner"
    )
    
    avg_review_score = reviews_sellers.groupby("seller_id")["review_score"].mean().rename("avg_review_score")
    review_count = reviews_sellers.groupby("seller_id")["review_score"].count().rename("review_count")

    seller_features = pd.concat([
        total_revenue, total_orders, avg_order_value, avg_delivery_time,
        late_rate, avg_freight_per_order, avg_processing_time,
        avg_review_score, review_count,
    ], axis=1).reset_index()
    
    seller_features.dropna(subset=["total_revenue", "total_orders"], inplace=True)
    
    numeric_cols = seller_features.select_dtypes(include="number").columns
    seller_features[numeric_cols] = seller_features[numeric_cols].fillna(
        seller_features[numeric_cols].median()
    )

    return seller_features


def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Clip features at 1st and 99th percentiles for robust scaling."""
    df_clean = df.copy()
    for col in columns:
        q01, q99 = df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)
    return df_clean


def train_revenue_model(features: pd.DataFrame, revenue_cols: list):
    """Train KMeans (k=2) explicitly targeting revenue concentration."""
    print("Training Revenue Clustering (k=2) ...")
    
    df_clean = handle_outliers(features, revenue_cols)
    X_revenue = df_clean[revenue_cols].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_revenue)

    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=15, max_iter=500, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    df_clean["rev_label"] = labels
    high_cluster_idx = df_clean.groupby("rev_label")["total_revenue"].mean().idxmax()

    segment_names = {
        high_cluster_idx: "High Revenue Segment",
        (1 - high_cluster_idx): "Low Revenue Segment"
    }

    return scaler, kmeans, segment_names


def train_behavior_model(features: pd.DataFrame, behavior_cols: list):
    """Train KMeans (k=3) focusing on delay metrics, reviews, and velocity."""
    print("Training Behavior Clustering (k=3) ...")
    
    df_behavior = handle_outliers(features, behavior_cols)
    X_behavior = df_behavior[behavior_cols].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_behavior)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X_scaled)

    df_behavior["beh_label"] = labels
    means = df_behavior.groupby("beh_label").mean(numeric_only=True)
    
    struggling_idx = means["late_delivery_rate"].idxmax()
    premium_idx = means["avg_order_value"].idxmax()
    
    efficient_idx = [i for i in [0, 1, 2] if i not in (struggling_idx, premium_idx)]
    efficient_idx = efficient_idx[0] if efficient_idx else 0

    segment_names = {
        efficient_idx: "Efficient Sellers",
        premium_idx: "Premium Sellers",
        struggling_idx: "Struggling Sellers"
    }

    return scaler, kmeans, segment_names


def main():
    df_orders, df_reviews = load_raw_data()
    seller_features = extract_features(df_orders, df_reviews)

    REVENUE_COLS = [
        "total_revenue", "total_orders", "avg_order_value", "avg_delivery_time_days", 
        "late_delivery_rate", "avg_freight_per_order", "avg_processing_time_days", 
        "avg_review_score", "review_count"
    ]
    
    BEHAVIOR_COLS = [
        "avg_order_value", "avg_delivery_time_days", "late_delivery_rate",
        "avg_freight_per_order", "avg_processing_time_days", "avg_review_score"
    ]

    scaler_rev, km_rev, rev_names = train_revenue_model(seller_features, REVENUE_COLS)
    scaler_beh, km_beh, beh_names = train_behavior_model(seller_features, BEHAVIOR_COLS)

    cluster_lookups = {
        "revenue_segments": rev_names,
        "behavior_segments": beh_names,
        "revenue_cols": REVENUE_COLS,
        "behavior_cols": BEHAVIOR_COLS,
        "revenue_bounds": {col: (seller_features[col].quantile(0.01), seller_features[col].quantile(0.99)) for col in REVENUE_COLS},
        "behavior_bounds": {col: (seller_features[col].quantile(0.01), seller_features[col].quantile(0.99)) for col in BEHAVIOR_COLS}
    }

    print(f"\nSaving artifacts to: {ARTIFACTS_DIR}")
    
    with open(os.path.join(ARTIFACTS_DIR, "scaler_revenue.pkl"), "wb") as f:
        pickle.dump(scaler_rev, f)
    with open(os.path.join(ARTIFACTS_DIR, "kmeans_revenue.pkl"), "wb") as f:
        pickle.dump(km_rev, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "scaler_behavior.pkl"), "wb") as f:
        pickle.dump(scaler_beh, f)
    with open(os.path.join(ARTIFACTS_DIR, "kmeans_behavior.pkl"), "wb") as f:
        pickle.dump(km_beh, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "cluster_lookups.pkl"), "wb") as f:
        pickle.dump(cluster_lookups, f)

    print("Done! Files:", os.listdir(ARTIFACTS_DIR))


if __name__ == "__main__":
    main()
