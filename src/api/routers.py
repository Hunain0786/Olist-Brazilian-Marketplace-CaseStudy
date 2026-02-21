from fastapi import APIRouter
from core.schemas import (
    HealthResponse,
    OrderInput,
    PredictionResponse,
    SellerBehaviorInput,
    SellerCategoryResponse,
    SellerRevenueInput,
)
from core.exceptions import ModelNotLoadedError
from core.predictor import DelayPredictor

predictor = DelayPredictor()

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        version="1.1.0",
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_delay(order: OrderInput):
    if not predictor.is_loaded:
        raise ModelNotLoadedError("Model artifacts not loaded.")
    result = predictor.predict(order)
    return PredictionResponse(**result)

@router.post(
    "/predict_revenue_category",
    response_model=SellerCategoryResponse,
    tags=["Seller Categorization"],
)
async def predict_revenue_category(seller: SellerRevenueInput):
    if not predictor.is_loaded:
        raise ModelNotLoadedError("Model artifacts not loaded.")
    result = predictor.predict_revenue_category(seller)
    return SellerCategoryResponse(**result)

@router.post(
    "/predict_behavior_category",
    response_model=SellerCategoryResponse,
    tags=["Seller Categorization"],
)
async def predict_behavior_category(seller: SellerBehaviorInput):
    if not predictor.is_loaded:
        raise ModelNotLoadedError("Model artifacts not loaded.")
    result = predictor.predict_behavior_category(seller)
    return SellerCategoryResponse(**result)
