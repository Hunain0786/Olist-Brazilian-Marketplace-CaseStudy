import os
import pickle
import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


STATE_TO_REGION = {
    'AC': 'North',        'AM': 'North',        'AP': 'North',    'PA': 'North',
    'RO': 'North',        'RR': 'North',        'TO': 'North',
    'AL': 'Northeast',    'BA': 'Northeast',    'CE': 'Northeast', 'MA': 'Northeast',
    'PB': 'Northeast',    'PE': 'Northeast',    'PI': 'Northeast', 'RN': 'Northeast',
    'SE': 'Northeast',
    'DF': 'Central-West', 'GO': 'Central-West', 'MT': 'Central-West', 'MS': 'Central-West',
    'ES': 'Southeast',    'MG': 'Southeast',    'RJ': 'Southeast', 'SP': 'Southeast',
    'PR': 'South',        'RS': 'South',        'SC': 'South',
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def extract_data_tables():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    
    return {
        "customers": pd.read_csv(os.path.join(path, "olist_customers_dataset.csv")),
        "items": pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv")),
        "reviews": pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv")),
        "orders": pd.read_csv(os.path.join(path, "olist_orders_dataset.csv")),
        "sellers": pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv")),
        "geolocation": pd.read_csv(os.path.join(path, "olist_geolocation_dataset.csv"))
    }


def compute_core_metrics(tables: dict) -> pd.DataFrame:
    orders, reviews = tables["orders"], tables["reviews"]
    
    df = pd.merge(orders, reviews, on='order_id')
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date', 'review_creation_date',
        'review_answer_timestamp',
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    delivered = df[df['order_status'] == 'delivered'].copy()
    delivered['actual_delivery_days'] = (
        delivered['order_delivered_customer_date'] - delivered['order_purchase_timestamp']
    ).dt.days
    delivered['estimated_delivery_days'] = (
        delivered['order_estimated_delivery_date'] - delivered['order_purchase_timestamp']
    ).dt.days
    delivered['delivery_delay_days'] = (
        delivered['actual_delivery_days'] - delivered['estimated_delivery_days']
    )
    return delivered


def join_geospatial_features(delivered: pd.DataFrame, tables: dict) -> pd.DataFrame:
    customers, items, sellers, geolocation = (
        tables["customers"], tables["items"], tables["sellers"], tables["geolocation"]
    )
    
    delivered = pd.merge(
        delivered,
        customers[['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']],
        on='customer_id', how='left',
    )
    order_sellers = items.groupby('order_id')['seller_id'].first().reset_index()
    order_sellers = pd.merge(
        order_sellers,
        sellers[['seller_id', 'seller_state', 'seller_zip_code_prefix', 'seller_city']],
        on='seller_id', how='left',
    )
    delivered = pd.merge(
        delivered,
        order_sellers[['order_id', 'seller_state', 'seller_zip_code_prefix', 'seller_city']],
        on='order_id', how='left',
    )
    delivered['order_size'] = delivered['order_id'].map(
        items.groupby('order_id')['order_item_id'].count()
    )

    zip_coords = (
        geolocation
        .groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']]
        .mean()
    )
    delivered['customer_lat'] = delivered['customer_zip_code_prefix'].map(zip_coords['geolocation_lat'])
    delivered['customer_lng'] = delivered['customer_zip_code_prefix'].map(zip_coords['geolocation_lng'])
    delivered['seller_lat']   = delivered['seller_zip_code_prefix'].map(zip_coords['geolocation_lat'])
    delivered['seller_lng']   = delivered['seller_zip_code_prefix'].map(zip_coords['geolocation_lng'])
    
    delivered['customer_seller_distance_km'] = haversine_km(
        delivered['customer_lat'], delivered['customer_lng'],
        delivered['seller_lat'],   delivered['seller_lng'],
    )
    
    median_dist = delivered['customer_seller_distance_km'].median()
    delivered['customer_seller_distance_km'] = delivered['customer_seller_distance_km'].fillna(median_dist)
    
    return delivered, zip_coords, median_dist


def build_training_features(delivered: pd.DataFrame) -> tuple:
    features_at_order_time = [
        'order_purchase_timestamp',
        'customer_state', 'customer_city', 'customer_zip_code_prefix',
        'seller_state',   'seller_city',   'seller_zip_code_prefix',
        'order_size', 'customer_seller_distance_km',
    ]
    model_df = delivered[features_at_order_time + ['delivery_delay_days']].copy()

    ts = pd.to_datetime(model_df['order_purchase_timestamp'])
    model_df['purchase_month']       = ts.dt.month
    model_df['purchase_day_of_week'] = ts.dt.dayofweek
    model_df['purchase_hour']        = ts.dt.hour
    model_df['is_weekend']           = (model_df['purchase_day_of_week'] >= 5).astype(int)
    model_df['purchase_quarter']     = ts.dt.quarter
    model_df = model_df.drop('order_purchase_timestamp', axis=1)

    model_df['state_pair'] = model_df['customer_state'] + '_TO_' + model_df['seller_state']
    model_df['customer_region'] = model_df['customer_state'].map(STATE_TO_REGION)
    model_df['seller_region']   = model_df['seller_state'].map(STATE_TO_REGION)
    model_df['region_pair']     = model_df['customer_region'] + '_TO_' + model_df['seller_region']

    overall_mean_delay = delivered['delivery_delay_days'].mean()
    city_mean_delay   = delivered.groupby('customer_city')['delivery_delay_days'].mean().to_dict()
    zip_mean_delay    = delivered.groupby('customer_zip_code_prefix')['delivery_delay_days'].mean().to_dict()
    zip_order_count   = delivered.groupby('customer_zip_code_prefix').size().to_dict()
    seller_zip_mean   = delivered.groupby('seller_zip_code_prefix')['delivery_delay_days'].mean().to_dict()

    clean_df = model_df.copy()
    clean_df['customer_city_avg_delay']    = clean_df['customer_city'].map(city_mean_delay).fillna(overall_mean_delay)
    clean_df['customer_zip_avg_delay']     = clean_df['customer_zip_code_prefix'].map(zip_mean_delay).fillna(overall_mean_delay)
    clean_df['customer_zip_order_volume']  = clean_df['customer_zip_code_prefix'].map(zip_order_count).fillna(0)
    clean_df['seller_zip_avg_delay']       = clean_df['seller_zip_code_prefix'].map(seller_zip_mean).fillna(overall_mean_delay)

    categorical_cols = [
        'customer_state', 'seller_state', 'state_pair',
        'customer_region', 'seller_region', 'region_pair',
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        clean_df[col + '_encoded'] = le.fit_transform(clean_df[col].astype(str))
        label_encoders[col] = le

    clean_df = clean_df.drop(
        columns=categorical_cols + [
            'customer_city', 'seller_city',
            'customer_zip_code_prefix', 'seller_zip_code_prefix',
        ]
    )
    clean_df = clean_df.dropna()
    
    lookups = {
        'overall_mean_delay': overall_mean_delay,
        'city_mean_delay': city_mean_delay,
        'zip_mean_delay': zip_mean_delay,
        'zip_order_count': zip_order_count,
        'seller_zip_mean': seller_zip_mean,
    }
    
    return clean_df, label_encoders, lookups


def run_training_pipeline(clean_df: pd.DataFrame):
    feature_cols = [c for c in clean_df.columns if c != 'delivery_delay_days']
    numerical_cols = [
        'purchase_month', 'purchase_day_of_week', 'purchase_hour',
        'purchase_quarter', 'customer_zip_order_volume',
        'customer_city_avg_delay', 'customer_zip_avg_delay', 'seller_zip_avg_delay',
        'customer_seller_distance_km',
    ]
    
    X = clean_df[feature_cols]
    y = clean_df['delivery_delay_days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_split=50, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nModel Performance:\nMAE: {mean_absolute_error(y_test, y_pred):.2f} days")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} days")
    print(f"R²: {r2_score(y_test, y_pred):.4f}\n")
    
    return model, scaler, feature_cols, numerical_cols


def main():
    tables = extract_data_tables()
    delivered = compute_core_metrics(tables)
    delivered, zip_coords, median_dist = join_geospatial_features(delivered, tables)
    
    clean_df, label_encoders, lookups = build_training_features(delivered)
    model, scaler, feature_cols, numerical_cols = run_training_pipeline(clean_df)

    lookup_tables = {
        **lookups,
        'median_distance': median_dist,
        'zip_coords': {
            "lat": zip_coords['geolocation_lat'].to_dict(),
            "lng": zip_coords['geolocation_lng'].to_dict(),
        },
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'state_to_region': STATE_TO_REGION,
    }

    with open(os.path.join(ARTIFACTS_DIR, 'rf_regressor.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(ARTIFACTS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(ARTIFACTS_DIR, 'lookup_tables.pkl'), 'wb') as f:
        pickle.dump(lookup_tables, f)

    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == '__main__':
    main()
