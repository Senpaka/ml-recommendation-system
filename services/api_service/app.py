import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from core.logging_config import setup_logging
from services.api_service.routers.health import router as health_router
from services.api_service.routers.recommendations import (
    router as recommendations_router,
)
from services.api_service.routers.root import router as root_router
from services.recommendations_service.recommendation import RecommendationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    setup_logging()

    artifacts_path = Path(__file__).resolve().parents[2] / "artifacts" / "current"

    logger.info("Loading recommendation artifacts from %s", artifacts_path)

    try:
        app.state.recommendation_service = RecommendationService.from_artifacts(
            artifacts_path
        )
    except Exception:
        logger.exception("Failed to load recommendation artifactts")
        raise

    logger.info("Recommendation service is ready")

    try:
        yield
    finally:
        logger.info("Application is shutting down")


app = FastAPI(
    version="0.0.1b",
    title="Recommendation system",
    description="API for recommendation system",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(root_router)
app.include_router(recommendations_router)
