
import logging
import os
import pickle
from typing import Any, List

import numpy as np
import pandas as pd
import shap

from core.exceptions import (
    ArtifactLoadError,
    FeatureEngineeringError,
    InvalidInputError,
    ModelNotLoadedError,
    PredictionInternalError,
)
from core.schemas import OrderInput, SellerBehaviorInput, SellerRevenueInput

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _load_pickle(name: str) -> Any:
    path = os.path.join(ARTIFACTS_DIR, name)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise ArtifactLoadError(name, FileNotFoundError(f"File not found at '{path}'"))
    except Exception as exc:
        raise ArtifactLoadError(name, exc) from exc


class DelayPredictor:
    """Predicts delivery delay using the notebook-trained model."""

    def __init__(self):
        # Delay model artifacts
        self.model: Any = None
        self.lookups: dict = {}
        self.explainer: Any = None

        # Seller clustering artifacts (unchanged)
        self.km_revenue: Any = None
        self.scaler_revenue: Any = None
        self.km_behavior: Any = None
        self.scaler_behavior: Any = None
        self.cluster_lookups: dict = {}

        self._loaded = False

    def load(self) -> None:
        logger.info("Loading model artifacts from '%s' …", ARTIFACTS_DIR)

        # ── Delay model (new artifacts from notebook) ────────────────────
        self.model   = _load_pickle("delay_model.pkl")
        self.lookups = _load_pickle("delay_lookup_tables.pkl")

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            raise ArtifactLoadError("shap_explainer", exc) from exc

        # ── Seller clustering (unchanged) ────────────────────────────────
        self.km_revenue      = _load_pickle("kmeans_revenue.pkl")
        self.scaler_revenue  = _load_pickle("scaler_revenue.pkl")
        self.km_behavior     = _load_pickle("kmeans_behavior.pkl")
        self.scaler_behavior = _load_pickle("scaler_behavior.pkl")
        self.cluster_lookups = _load_pickle("cluster_lookups.pkl")

        self._loaded = True
        logger.info("All artifacts loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Validation ────────────────────────────────────────────────────────

    @staticmethod
    def _validate_order(order: OrderInput) -> None:
        if order.total_order_items < 1:
            raise InvalidInputError("total_order_items", "must be at least 1")

        state_codes = {
            "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
            "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
            "RO", "RR", "RS", "SC", "SE", "SP", "TO",
        }
        for field, val in (("customer_state", order.customer_state),
                           ("seller_state",   order.seller_state)):
            if val.upper() not in state_codes:
                raise InvalidInputError(field, f"'{val}' is not a recognised Brazilian state code")

    @staticmethod
    def _validate_revenue_input(seller: SellerRevenueInput) -> None:
        if seller.total_revenue < 0:
            raise InvalidInputError("total_revenue", "must be >= 0")
        if seller.total_orders < 1:
            raise InvalidInputError("total_orders", "must be at least 1")
        if seller.avg_order_value < 0:
            raise InvalidInputError("avg_order_value", "must be >= 0")
        if not (0.0 <= seller.late_delivery_rate <= 1.0):
            raise InvalidInputError("late_delivery_rate", "must be between 0.0 and 1.0")
        if seller.avg_review_score < 1.0 or seller.avg_review_score > 5.0:
            raise InvalidInputError("avg_review_score", "must be between 1.0 and 5.0")
        if seller.review_count < 0:
            raise InvalidInputError("review_count", "must be >= 0")
        if seller.avg_freight_per_order < 0:
            raise InvalidInputError("avg_freight_per_order", "must be >= 0")

    @staticmethod
    def _validate_behavior_input(seller: SellerBehaviorInput) -> None:
        if seller.avg_order_value < 0:
            raise InvalidInputError("avg_order_value", "must be >= 0")
        if not (0.0 <= seller.late_delivery_rate <= 1.0):
            raise InvalidInputError("late_delivery_rate", "must be between 0.0 and 1.0")
        if seller.avg_review_score < 1.0 or seller.avg_review_score > 5.0:
            raise InvalidInputError("avg_review_score", "must be between 1.0 and 5.0")
        if seller.avg_freight_per_order < 0:
            raise InvalidInputError("avg_freight_per_order", "must be >= 0")

    # ── Feature engineering (mirrors notebook add_features) ───────────────

    def _build_features(self, order: OrderInput) -> pd.DataFrame:
        """Replicate the notebook feature engineering for a single order."""
        lk = self.lookups
        medians = lk.get("medians", {})

        # --- Helper: look up a stat from a {col: [values]} dict-of-lists ---
        def _lookup(stats_key: str, match_col: str, value_col: str, match_val):
            stats = lk.get(stats_key, {})
            keys = stats.get(match_col, [])
            vals = stats.get(value_col, [])
            try:
                idx = keys.index(match_val)
                return vals[idx]
            except (ValueError, IndexError):
                return medians.get(value_col, 0.0)

        # Seller-level aggregates
        seller_total_orders = _lookup("seller_stats", "seller_id", "seller_total_orders", order.seller_id)
        seller_avg_freight  = _lookup("seller_stats", "seller_id", "seller_avg_freight",  order.seller_id)
        seller_avg_delay    = _lookup("seller_stats", "seller_id", "seller_avg_delay",    order.seller_id)
        seller_delay_std    = _lookup("seller_stats", "seller_id", "seller_delay_std",    order.seller_id)

        # State / city / category average delays
        avg_delay_state    = _lookup("state_stats", "customer_state",                "avg_delay_state",    order.customer_state)
        avg_delay_city     = _lookup("city_stats",  "customer_city",                 "avg_delay_city",     order.customer_city)
        avg_delay_category = _lookup("cat_stats",   "product_category_name_english", "avg_delay_category", order.product_category_name_english)

        # Order physical properties
        total_order_weight       = order.product_weight_g * order.total_order_items
        total_order_items        = order.total_order_items
        max_product_height       = order.product_height_cm
        max_product_width        = order.product_width_cm
        max_product_length       = order.product_length_cm
        volumetric_weight        = (order.product_length_cm * order.product_width_cm * order.product_height_cm) / 5000
        total_volumetric_weight  = volumetric_weight * order.total_order_items

        # Time features
        ts = order.order_purchase_timestamp
        purchase_dayofweek  = ts.weekday()
        purchase_month      = ts.month
        purchase_hour       = ts.hour
        purchase_is_weekend = int(purchase_dayofweek >= 5)

        # Promised days
        promised_days = (order.order_estimated_delivery_date - order.order_purchase_timestamp).total_seconds() / 86400

        # Same state / city flags
        same_state = int(order.customer_state.upper() == order.seller_state.upper())
        same_city  = int(order.customer_city.lower()  == order.seller_city.lower())

        # Build the row in the same column order the model was trained on
        row = {
            'seller_total_orders':     seller_total_orders,
            'seller_avg_freight':      seller_avg_freight,
            'seller_avg_delay':        seller_avg_delay,
            'seller_delay_std':        seller_delay_std,
            'avg_delay_state':         avg_delay_state,
            'avg_delay_city':          avg_delay_city,
            'avg_delay_category':      avg_delay_category,
            'total_order_weight':      total_order_weight,
            'total_order_items':       total_order_items,
            'total_volumetric_weight': total_volumetric_weight,
            'max_product_height':      max_product_height,
            'max_product_width':       max_product_width,
            'max_product_length':      max_product_length,
            'freight_value':           order.freight_value,
            'promised_days':           promised_days,
            'purchase_dayofweek':      purchase_dayofweek,
            'purchase_month':          purchase_month,
            'purchase_hour':           purchase_hour,
            'purchase_is_weekend':     purchase_is_weekend,
            'same_state':              same_state,
            'same_city':               same_city,
            'price':                   order.price,
        }

        feature_cols = lk.get("feature_cols", list(row.keys()))
        df = pd.DataFrame([row], columns=feature_cols)

        # Fill any remaining NaN with train medians
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(medians.get(col, 0.0))

        return df

    # ── SHAP explanations ─────────────────────────────────────────────────

    def _get_shap_influences(self, df: pd.DataFrame, top_n: int = 5) -> List[dict]:
        try:
            shap_values   = self.explainer.shap_values(df)
            feature_names = list(df.columns)
            values        = shap_values[0]

            indexed = sorted(
                zip(feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            return [
                {
                    "feature":    name,
                    "shap_value": round(float(val), 4),
                    "direction":  "increases delay" if val > 0 else "decreases delay",
                }
                for name, val in indexed[:top_n]
            ]
        except Exception as exc:
            logger.warning("SHAP explanation failed (returning empty list): %s", exc)
            return []

    # ── Predict delivery delay ────────────────────────────────────────────

    def predict(self, order: OrderInput) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Call load() before predict().")

        self._validate_order(order)

        try:
            df = self._build_features(order)
        except (FeatureEngineeringError, InvalidInputError):
            raise
        except Exception as exc:
            raise PredictionInternalError(f"Unexpected feature error: {exc}") from exc

        try:
            prediction = float(self.model.predict(df)[0])
        except Exception as exc:
            raise PredictionInternalError(f"Model inference failed: {exc}") from exc

        top_features = self._get_shap_influences(df)

        if prediction > 7:
            category = "Severely Late"
        elif prediction > 3:
            category = "Moderately Late"
        elif prediction > 0:
            category = "Slightly Late"
        elif prediction > -3:
            category = "On Time"
        else:
            category = "Early"

        return {
            "predicted_delay_days": round(prediction, 2),
            "delay_category":       category,
            "confidence_note": (
                "Based on historical patterns for this seller, route, "
                "product dimensions, and time of purchase."
            ),
            "top_features": top_features,
        }

    # ── Seller clustering (unchanged) ─────────────────────────────────────

    def predict_revenue_category(self, seller: SellerRevenueInput) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Call load() before predict_revenue_category().")

        self._validate_revenue_input(seller)

        try:
            cols   = self.cluster_lookups["revenue_cols"]
            bounds = self.cluster_lookups["revenue_bounds"]

            row = seller.dict()
            df  = pd.DataFrame([row])

            for col in cols:
                q01, q99 = bounds[col]
                df[col]  = df[col].clip(lower=q01, upper=q99)

            df       = df[cols]
            X_scaled = self.scaler_revenue.transform(df.values)
        except (InvalidInputError, KeyError) as exc:
            raise FeatureEngineeringError(str(exc)) from exc
        except Exception as exc:
            raise PredictionInternalError(f"Revenue feature prep failed: {exc}") from exc

        try:
            cluster_id   = int(self.km_revenue.predict(X_scaled)[0])
        except Exception as exc:
            raise PredictionInternalError(f"Revenue model inference failed: {exc}") from exc

        segment_name = self.cluster_lookups["revenue_segments"].get(
            cluster_id, "Unknown Segment"
        )
        logger.debug("Revenue cluster %d → '%s'", cluster_id, segment_name)

        return {"cluster_id": cluster_id, "segment_name": segment_name}

    def predict_behavior_category(self, seller: SellerBehaviorInput) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Call load() before predict_behavior_category().")

        self._validate_behavior_input(seller)

        try:
            cols   = self.cluster_lookups["behavior_cols"]
            bounds = self.cluster_lookups["behavior_bounds"]

            row = seller.dict()
            df  = pd.DataFrame([row])

            for col in cols:
                q01, q99 = bounds[col]
                df[col]  = df[col].clip(lower=q01, upper=q99)

            df       = df[cols]
            X_scaled = self.scaler_behavior.transform(df.values)
        except (InvalidInputError, KeyError) as exc:
            raise FeatureEngineeringError(str(exc)) from exc
        except Exception as exc:
            raise PredictionInternalError(f"Behavior feature prep failed: {exc}") from exc

        try:
            cluster_id = int(self.km_behavior.predict(X_scaled)[0])
        except Exception as exc:
            raise PredictionInternalError(f"Behavior model inference failed: {exc}") from exc

        segment_name = self.cluster_lookups["behavior_segments"].get(
            cluster_id, "Unknown Segment"
        )
        logger.debug("Behavior cluster %d → '%s'", cluster_id, segment_name)

        return {"cluster_id": cluster_id, "segment_name": segment_name}
