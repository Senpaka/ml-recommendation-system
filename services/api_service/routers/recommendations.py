from typing import Annotated

from fastapi import APIRouter, Path, Query, Request

from services.api_service.schemas import RecommendationItem
from services.recommendations_service.recommendation import RecommendationService

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


@router.get("/batch")
def get_batch_recommendations(
    request: Request,
    user_ids: Annotated[list[int], Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[int, list[dict[str, int | float]]]:

    service: RecommendationService = request.app.state.recommendation_service

    return service.recommend_batch(user_ids=user_ids, k=k)


@router.get("/{user_id}", response_model=list[RecommendationItem])
def get_recommendations(
    request: Request,
    user_id: Annotated[int, Path(gt=0)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[dict[str, int | float]]:

    service: RecommendationService = request.app.state.recommendation_service

    return service.recommend(user_id=user_id, k=k)
