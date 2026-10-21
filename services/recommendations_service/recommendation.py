import logging
from pathlib import Path
from typing import Any, Self

import numpy as np
from catboost import Pool

from src.artifacts import load_artifacts
from src.features.features import build_features

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self, artifacts: dict[str, Any]) -> None:
        self.artifacts = artifacts

    @classmethod
    def from_artifacts(cls, path: str | Path) -> Self:

        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"Artifacts directory not found {path}")

        return cls(artifacts=load_artifacts(str(path)))

    def recommend(self, user_id: int, k: int = 10) -> list[dict[str, int | float]]:
        return self.recommend_batch(user_ids=[user_id], k=k)[user_id]

    def recommend_batch(
        self, user_ids: list[int], k: int = 10
    ) -> dict[int, list[dict[str, int | float]]]:

        if k <= 0:
            logger.error("Generating is dropped. k must be positive.")
            raise ValueError("k must be positive.")

        user_ids = list(dict.fromkeys(user_ids))

        results: dict[int, list[dict[str, int | float]]] = {
            user_id: [] for user_id in user_ids
        }

        if not user_ids:
            logger.warning("No one users for recommend")
            return results

        logger.debug(f"Generating recommendations: user_count={len(user_ids)}, k={k}")

        retrieval = self.artifacts["retrieval"]
        warm_user_ids: list[int] = []

        for user_id in user_ids:
            if retrieval.can_personalize(user_id):
                warm_user_ids.append(user_id)
                continue

            logger.debug(f"Using popular recommendations: user_id={user_id}")

            recommendations = retrieval.recommend_popular(user_id=user_id, k=k)

            results[user_id] = [
                {
                    "movie_id": int(movie_id),
                    "score": float(score),
                }
                for movie_id, score in recommendations
            ]

        if not warm_user_ids:
            return results

        logger.info(
            f"Using full pipeline recommendations: user_count={len(warm_user_ids)}"
        )

        candidates = retrieval.make_candidates(users=warm_user_ids)

        if candidates.empty:
            logger.warning(
                f"No recommendations candidates: user_count={len(warm_user_ids)}"
            )
            return results

        features = build_features(
            candidates=candidates,
            history=self.artifacts["history"],
            df_movies=self.artifacts["movies"],
            df_users=self.artifacts["users"],
            query_prefix="inference",
        )

        metadata = self.artifacts["metadata"]

        pool = Pool(
            data=features[metadata["feature_columns"]],
            cat_features=metadata["cat_features"],
        )

        scored = features[["user_id", "movie_id"]].copy()
        scored["score"] = self.artifacts["ranker"].predict(pool)

        top = (
            scored.sort_values(
                ["user_id", "score", "movie_id"], ascending=[True, False, True]
            )
            .groupby("user_id", sort=False)
            .head(k)
        )

        logger.info("Recommendations is ready.")

        for user_id_value, movie_id_value, score_value in zip(
            top["user_id"].to_numpy(dtype=np.int64),
            top["movie_id"].to_numpy(dtype=np.int64),
            top["score"].to_numpy(dtype=np.float64),
            strict=True,
        ):
            results[int(user_id_value)].append(
                {
                    "movie_id": int(movie_id_value),
                    "score": float(score_value),
                }
            )

        logger.info(
            "Recommendations ready: users=%s, personalized=%s, cold=%s",
            len(user_ids),
            len(warm_user_ids),
            len(user_ids) - len(warm_user_ids),
        )

        return results
