import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import seaborn as sns


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

EXPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)



def extract_data_tables() -> dict:
    """Download the Olist dataset from Kaggle and load all required tables."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print("Path to dataset files:", path)

    return {
        "customers":            pd.read_csv(os.path.join(path, 'olist_customers_dataset.csv')),
        "items":                pd.read_csv(os.path.join(path, 'olist_order_items_dataset.csv')),
        "payments":             pd.read_csv(os.path.join(path, 'olist_order_payments_dataset.csv')),
        "reviews":              pd.read_csv(os.path.join(path, 'olist_order_reviews_dataset.csv')),
        "orders":               pd.read_csv(os.path.join(path, 'olist_orders_dataset.csv')),
        "products":             pd.read_csv(os.path.join(path, 'olist_products_dataset.csv')),
        "sellers":              pd.read_csv(os.path.join(path, 'olist_sellers_dataset.csv')),
        "category_translation": pd.read_csv(os.path.join(path, 'product_category_name_translation.csv')),
    }



def build_core_dataframe(tables: dict) -> pd.DataFrame:
    """Merge tables into a single flat DataFrame and compute the target variable."""
    orders     = tables["orders"].copy()
    items      = tables["items"]
    products   = tables["products"]
    customers  = tables["customers"]
    sellers    = tables["sellers"]
    cat_trans  = tables["category_translation"]

    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date',
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col])

    products = products.merge(cat_trans, on='product_category_name', how='left')

    df = (
        items
        .merge(products,   on='product_id',  how='left')
        .merge(orders,     on='order_id',    how='left')
        .merge(customers,  on='customer_id', how='left')
        .merge(sellers,    on='seller_id',   how='left')
    )

    df = df[df['order_status'] == 'delivered'].copy()

    # Drop rows where key timestamps are missing
    df = df.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])

    # ── Target Variable ───────────────────────────────────────────────────────
    df['delay_delivery_days'] = (
        df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
    ).dt.total_seconds() / 86400   # fractional days for more resolution

    return df



def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data chronologically so no future information leaks into training."""
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test



def add_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compute aggregation features on train_df, then map them onto both splits.
    This prevents target leakage from the test set into the statistics.
    """

    # --- 4a. Seller-level aggregates ---
    seller_stats = (
        train_df.groupby('seller_id')
        .agg(
            seller_total_orders = ('order_id', 'nunique'),
            seller_avg_freight  = ('freight_value', 'mean'),
            seller_avg_delay    = ('delay_delivery_days', 'mean'),
            seller_delay_std    = ('delay_delivery_days', 'std'),
        )
        .reset_index()
    )

    # --- 4b. Customer-state aggregates ---
    state_stats = (
        train_df.groupby('customer_state')
        .agg(avg_delay_state=('delay_delivery_days', 'mean'))
        .reset_index()
    )

    city_stats = (
        train_df.groupby('customer_city')
        .agg(avg_delay_city=('delay_delivery_days', 'mean'))
        .reset_index()
    )

    cat_stats = (
        train_df.groupby('product_category_name_english')
        .agg(avg_delay_category=('delay_delivery_days', 'mean'))
        .reset_index()
    )

    for split in [train_df, test_df]:
        split['total_order_weight']  = split.groupby('order_id')['product_weight_g'].transform('sum')
        split['total_order_items']   = split.groupby('order_id')['order_item_id'].transform('count')
        split['max_product_height']  = split.groupby('order_id')['product_height_cm'].transform('max')
        split['max_product_width']   = split.groupby('order_id')['product_width_cm'].transform('max')
        split['max_product_length']  = split.groupby('order_id')['product_length_cm'].transform('max')

        split['volumetric_weight'] = (
            split['product_length_cm']
            * split['product_width_cm']
            * split['product_height_cm'] / 5000
        )
        split['total_volumetric_weight'] = split.groupby('order_id')['volumetric_weight'].transform('sum')

    for split in [train_df, test_df]:
        ts = split['order_purchase_timestamp']
        split['purchase_dayofweek']  = ts.dt.dayofweek
        split['purchase_month']      = ts.dt.month
        split['purchase_hour']       = ts.dt.hour
        split['purchase_is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)

        split['promised_days'] = (
            split['order_estimated_delivery_date'] - split['order_purchase_timestamp']
        ).dt.total_seconds() / 86400

        # Distance proxy: same-state / same-city 
        split['same_state'] = (split['customer_state'] == split['seller_state']).astype(int)
        split['same_city']  = (split['customer_city']  == split['seller_city']).astype(int)

    # --- 4g. Merge stats into both splits ---
    train_df = train_df.merge(seller_stats, on='seller_id',                       how='left')
    test_df  = test_df.merge(seller_stats,  on='seller_id',                       how='left')

    train_df = train_df.merge(state_stats, on='customer_state',                   how='left')
    test_df  = test_df.merge(state_stats,  on='customer_state',                   how='left')

    train_df = train_df.merge(city_stats, on='customer_city',                     how='left')
    test_df  = test_df.merge(city_stats,  on='customer_city',                     how='left')

    train_df = train_df.merge(cat_stats, on='product_category_name_english',      how='left')
    test_df  = test_df.merge(cat_stats,  on='product_category_name_english',      how='left')

    feature_lookups = {
        'seller_stats':  seller_stats,
        'state_stats':   state_stats,
        'city_stats':    city_stats,
        'cat_stats':     cat_stats,
    }

    return train_df, test_df, feature_lookups



FEATURE_COLS = [
    'seller_total_orders',
    'seller_avg_freight',
    'seller_avg_delay',
    'seller_delay_std',
    'avg_delay_state',
    'avg_delay_city',
    'avg_delay_category',
    'total_order_weight',
    'total_order_items',
    'total_volumetric_weight',
    'max_product_height',
    'max_product_width',
    'max_product_length',
    'freight_value',
    'promised_days',
    'purchase_dayofweek',
    'purchase_month',
    'purchase_hour',
    'purchase_is_weekend',
    'same_state',
    'same_city',
    # Price
    'price',
]

TARGET = 'delay_delivery_days'


def prepare_train_test(train: pd.DataFrame, test: pd.DataFrame):
    """Select features, fill NaN with train medians, clip target, and split X/y."""
    train_clean = train[FEATURE_COLS + [TARGET]].copy()
    test_clean  = test[FEATURE_COLS  + [TARGET]].copy()

    medians = train_clean.median()
    train_clean = train_clean.fillna(medians)
    test_clean  = test_clean.fillna(medians)  # use train medians for test

    # Clip extreme target values
    train_clean[TARGET] = train_clean[TARGET].clip(-30, 30)
    test_clean[TARGET]  = test_clean[TARGET].clip(-30, 30)

    X_train, y_train = train_clean[FEATURE_COLS], train_clean[TARGET]
    X_test,  y_test  = test_clean[FEATURE_COLS],  test_clean[TARGET]

    print(f"Features: {len(FEATURE_COLS)}  |  Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    return X_train, y_train, X_test, y_test, medians



def train_models(X_train, y_train, X_test, y_test):
    """Train RandomForest and XGBoost/GradientBoosting, return the best model."""

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.6,
        random_state=42,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBRegressor
        gbm = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        gbm_name = "XGBoost"
    except ImportError:
        gbm = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=4,
            random_state=42,
        )
        gbm_name = "GradientBoosting"

    models = {"RandomForest": rf, gbm_name: gbm}

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "preds": preds}
        print(f"  RMSE : {rmse:.4f} days")
        print(f"  MAE  : {mae:.4f} days")
        print(f"  R²   : {r2:.4f}")

    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name}  (R² = {results[best_model_name]['R2']:.4f})")

    return models, results, best_model_name, best_model



def save_diagnostics(best_model, best_model_name, results, y_test, train_clean_df):
    """Generate and save model-diagnostics chart and correlation heatmap."""

    best_preds = results[best_model_name]["preds"]
    fi = pd.Series(
        best_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    fi.head(15).sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title(f"Top 15 Feature Importances ({best_model_name})")
    axes[0].set_xlabel("Importance")

    axes[1].scatter(y_test, best_preds, alpha=0.3, s=10, color='steelblue')
    lim = (
        min(y_test.min(), best_preds.min()) - 2,
        max(y_test.max(), best_preds.max()) + 2,
    )
    axes[1].plot(lim, lim, 'r--', linewidth=1)
    axes[1].set_xlim(lim)
    axes[1].set_ylim(lim)
    axes[1].set_xlabel("Actual delay (days)")
    axes[1].set_ylabel("Predicted delay (days)")
    axes[1].set_title(
        f"Actual vs Predicted ({best_model_name})\n"
        f"R²={results[best_model_name]['R2']:.3f}  RMSE={results[best_model_name]['RMSE']:.3f}"
    )

    plt.tight_layout()
    diag_path = os.path.join(EXPORTS_DIR, 'model_diagnostics.png')
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f"Diagnostics chart saved → {diag_path}")

    plt.figure(figsize=(14, 11))
    corr = train_clean_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.3)
    plt.title("Feature Correlation Matrix (train set)")
    plt.tight_layout()
    heatmap_path = os.path.join(EXPORTS_DIR, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Correlation heatmap saved → {heatmap_path}")

    plt.figure(figsize=(15, 8))
    sns.heatmap(train_clean_df.corr(), annot=True, cmap='YlGnBu')
    plt.tight_layout()
    full_heatmap_path = os.path.join(EXPORTS_DIR, 'correlation_heatmap_annotated.png')
    plt.savefig(full_heatmap_path, dpi=150)
    plt.close()
    print(f"Annotated heatmap saved → {full_heatmap_path}")



def save_artifacts(best_model, best_model_name, medians, feature_lookups):
    """Persist model and supporting lookup tables for the inference API."""

    lookup_tables = {
        'feature_cols':   FEATURE_COLS,
        'target':         TARGET,
        'medians':        medians.to_dict(),
        'seller_stats':   feature_lookups['seller_stats'].to_dict(orient='list'),
        'state_stats':    feature_lookups['state_stats'].to_dict(orient='list'),
        'city_stats':     feature_lookups['city_stats'].to_dict(orient='list'),
        'cat_stats':      feature_lookups['cat_stats'].to_dict(orient='list'),
        'best_model_name': best_model_name,
    }

    with open(os.path.join(ARTIFACTS_DIR, 'delay_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(ARTIFACTS_DIR, 'delay_lookup_tables.pkl'), 'wb') as f:
        pickle.dump(lookup_tables, f)

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print(f"  - delay_model.pkl  ({best_model_name})")
    print(f"  - delay_lookup_tables.pkl")


# ── 9. Main ───────────────────────────────────────────────────────────────────

def main():
    tables = extract_data_tables()

    df = build_core_dataframe(tables)

    train, test = time_based_split(df)

    train, test, feature_lookups = add_features(train, test)

    X_train, y_train, X_test, y_test, medians = prepare_train_test(train, test)

    models, results, best_model_name, best_model = train_models(
        X_train, y_train, X_test, y_test
    )

    train_clean_df = train[FEATURE_COLS + [TARGET]].fillna(medians)
    train_clean_df[TARGET] = train_clean_df[TARGET].clip(-30, 30)
    save_diagnostics(best_model, best_model_name, results, y_test, train_clean_df)

    save_artifacts(best_model, best_model_name, medians, feature_lookups)


if __name__ == '__main__':
    main()
