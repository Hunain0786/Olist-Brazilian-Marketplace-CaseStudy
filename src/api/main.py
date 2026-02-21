import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.exceptions import ArtifactLoadError
from api.handlers import register_exception_handlers
from api.routers import predictor, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor.load()
        logger.info("✅ All model artifacts loaded successfully.")
    except ArtifactLoadError as exc:
        logger.error("⚠️  Artifact load error: %s", exc)
        logger.error("Run `python src/train_model.py` and `python src/train_seller_models.py` first.")
    except Exception as exc:
        logger.exception("⚠️  Unexpected error during startup: %s", exc)
    yield


app = FastAPI(
    title="Delivery Delay & Seller Category Predictor",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)
