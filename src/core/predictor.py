
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



def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class DelayPredictor:

    def __init__(self):
        self.model: Any = None
        self.scaler: Any = None
        self.label_encoders: dict = {}
        self.lookups: dict = {}
        self.explainer: Any = None

        self.km_revenue: Any = None
        self.scaler_revenue: Any = None
        self.km_behavior: Any = None
        self.scaler_behavior: Any = None
        self.cluster_lookups: dict = {}

        self._loaded = False


    def load(self) -> None:
        logger.info("Loading model artifacts from '%s' …", ARTIFACTS_DIR)

        self.model          = _load_pickle("rf_regressor.pkl")
        self.scaler         = _load_pickle("scaler.pkl")
        self.label_encoders = _load_pickle("label_encoders.pkl")
        self.lookups        = _load_pickle("lookup_tables.pkl")

        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            raise ArtifactLoadError("shap_explainer", exc) from exc

        self.km_revenue     = _load_pickle("kmeans_revenue.pkl")
        self.scaler_revenue = _load_pickle("scaler_revenue.pkl")
        self.km_behavior    = _load_pickle("kmeans_behavior.pkl")
        self.scaler_behavior= _load_pickle("scaler_behavior.pkl")
        self.cluster_lookups= _load_pickle("cluster_lookups.pkl")

        self._loaded = True
        logger.info("All artifacts loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

 
    @staticmethod
    def _validate_order(order: OrderInput) -> None:
        if order.order_size < 1:
            raise InvalidInputError("order_size", "must be at least 1")

        state_codes = {
            "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA",
            "MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN",
            "RO","RR","RS","SC","SE","SP","TO",
        }
        for field, val in (("customer_state", order.customer_state),
                           ("seller_state",   order.seller_state)):
            if val.upper() not in state_codes:
                raise InvalidInputError(field, f"'{val}' is not a recognised Brazilian state code")

        if order.customer_zip_code_prefix < 1000 or order.customer_zip_code_prefix > 99999:
            raise InvalidInputError("customer_zip_code_prefix", "must be a 5-digit integer (10001–99999)")
        if order.seller_zip_code_prefix < 1000 or order.seller_zip_code_prefix > 99999:
            raise InvalidInputError("seller_zip_code_prefix", "must be a 5-digit integer (10001–99999)")

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

    def _build_features(self, order: OrderInput) -> pd.DataFrame:
        lk = self.lookups

        try:
            state_to_region = lk["state_to_region"]
        except KeyError:
            raise FeatureEngineeringError("lookup table is missing 'state_to_region'")

        ts = order.order_purchase_timestamp
        purchase_month       = ts.month
        purchase_day_of_week = ts.weekday()
        purchase_hour        = ts.hour
        is_weekend           = int(purchase_day_of_week >= 5)
        purchase_quarter     = (ts.month - 1) // 3 + 1

      
        try:
            zip_lat = lk["zip_coords"]["lat"]
            zip_lng = lk["zip_coords"]["lng"]
        except KeyError:
            raise FeatureEngineeringError("lookup table is missing 'zip_coords'")

        c_lat = zip_lat.get(order.customer_zip_code_prefix)
        c_lng = zip_lng.get(order.customer_zip_code_prefix)
        s_lat = zip_lat.get(order.seller_zip_code_prefix)
        s_lng = zip_lng.get(order.seller_zip_code_prefix)

        if None in (c_lat, c_lng, s_lat, s_lng):
            distance_km = lk.get("median_distance", 500.0)
            logger.debug(
                "Zip code(s) not found in lookup; using median distance %.1f km.", distance_km
            )
        else:
            try:
                distance_km = float(_haversine_km(c_lat, c_lng, s_lat, s_lng))
            except Exception as exc:
                raise FeatureEngineeringError(f"Haversine calculation failed: {exc}") from exc

   
        overall = lk.get("overall_mean_delay", 0.0)
        customer_city_avg_delay   = lk["city_mean_delay"].get(order.customer_city, overall)
        customer_zip_avg_delay    = lk["zip_mean_delay"].get(order.customer_zip_code_prefix, overall)
        customer_zip_order_volume = lk["zip_order_count"].get(order.customer_zip_code_prefix, 0)
        seller_zip_avg_delay      = lk["seller_zip_mean"].get(order.seller_zip_code_prefix, overall)

       
        customer_region = state_to_region.get(order.customer_state, "Unknown")
        seller_region   = state_to_region.get(order.seller_state, "Unknown")
        state_pair      = f"{order.customer_state}_TO_{order.seller_state}"
        region_pair     = f"{customer_region}_TO_{seller_region}"

        def _safe_encode(col_name: str, value: str) -> int:
            le = self.label_encoders.get(col_name)
            if le is None:
                logger.warning("No LabelEncoder found for '%s'; defaulting to 0.", col_name)
                return 0
            if value in le.classes_:
                return int(le.transform([value])[0])
            logger.debug("Unseen label '%s' for '%s'; defaulting to 0.", value, col_name)
            return 0

        row = {
            "order_size":                   order.order_size,
            "customer_seller_distance_km":  distance_km,
            "purchase_month":               purchase_month,
            "purchase_day_of_week":         purchase_day_of_week,
            "purchase_hour":                purchase_hour,
            "is_weekend":                   is_weekend,
            "purchase_quarter":             purchase_quarter,
            "customer_city_avg_delay":      customer_city_avg_delay,
            "customer_zip_avg_delay":       customer_zip_avg_delay,
            "customer_zip_order_volume":    customer_zip_order_volume,
            "seller_zip_avg_delay":         seller_zip_avg_delay,
            "customer_state_encoded":       _safe_encode("customer_state", order.customer_state),
            "seller_state_encoded":         _safe_encode("seller_state", order.seller_state),
            "state_pair_encoded":           _safe_encode("state_pair", state_pair),
            "customer_region_encoded":      _safe_encode("customer_region", customer_region),
            "seller_region_encoded":        _safe_encode("seller_region", seller_region),
            "region_pair_encoded":          _safe_encode("region_pair", region_pair),
        }

        try:
            df = pd.DataFrame([row], columns=lk["feature_cols"])
        except KeyError:
            raise FeatureEngineeringError("lookup table is missing 'feature_cols'")

        return df

    def _get_shap_influences(self, df: pd.DataFrame, top_n: int = 5) -> List[dict]:
        try:
            shap_values  = self.explainer.shap_values(df)
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
            num_cols   = self.lookups["numerical_cols"]
            df[num_cols] = self.scaler.transform(df[num_cols])
            prediction   = float(self.model.predict(df)[0])
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
                "Based on historical patterns for this customer–seller route, "
                "order size, and time of purchase."
            ),
            "top_features": top_features,
        }

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
