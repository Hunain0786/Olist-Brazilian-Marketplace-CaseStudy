from fastapi import Request
from fastapi.responses import JSONResponse

from exceptions import (
    ArtifactLoadError,
    FeatureEngineeringError,
    InvalidInputError,
    ModelNotLoadedError,
    PredictionInternalError,
)

import logging
logger = logging.getLogger(__name__)

async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    logger.warning("503 Model not loaded: %s %s", request.method, request.url)
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_loaded",
            "detail": str(exc),
        },
    )

async def invalid_input_handler(request: Request, exc: InvalidInputError):
    logger.info("400 Invalid input [%s]: %s", exc.field, exc)
    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_input",
            "field": exc.field,
            "detail": str(exc),
        },
    )

async def feature_engineering_handler(request: Request, exc: FeatureEngineeringError):
    logger.error("500 Feature engineering error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "feature_engineering_error", "detail": str(exc)},
    )

async def prediction_internal_handler(request: Request, exc: PredictionInternalError):
    logger.error("500 Internal prediction error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "prediction_internal_error", "detail": str(exc)},
    )

async def artifact_load_handler(request: Request, exc: ArtifactLoadError):
    logger.error("500 Artifact load error (%s): %s", exc.artifact, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "artifact_load_error",
            "artifact": exc.artifact,
            "detail": str(exc),
        },
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("500 Unhandled exception: %s %s → %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
        },
    )

def register_exception_handlers(app):
    app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
    app.add_exception_handler(InvalidInputError, invalid_input_handler)
    app.add_exception_handler(FeatureEngineeringError, feature_engineering_handler)
    app.add_exception_handler(PredictionInternalError, prediction_internal_handler)
    app.add_exception_handler(ArtifactLoadError, artifact_load_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
