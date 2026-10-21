from pathlib import Path

from catboost import Pool

from src.artifacts import load_artifacts
from src.features.features import build_features


class RecommendationService:
    def __init__(self, artifacts: dict):
        self.artifacts = artifacts

    @classmethod
    def from_artifacts(cls, path: str | Path):
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"Artifacts directory not found {path}")

        return cls(artifacts=load_artifacts(str(path)))

    def recommend(self, user_id: int, k: int = 10):

        if k <= 0:
            raise ValueError("k must be positive.")

        ranker = self.artifacts["ranker"]
        retrieval = self.artifacts["retrieval"]
        metadata = self.artifacts["metadata"]

        if not retrieval.can_personalize(user_id):
            recommendations = retrieval.recommend_popular(user_id=user_id, k=k)

            return [
                {
                    "movie_id": int(movie_id),
                    "score": float(score),
                }
                for movie_id, score in recommendations
            ]

        candidates = retrieval.make_candidates(users=[user_id])

        if candidates.empty:
            return []

        features = build_features(
            candidates=candidates,
            history=self.artifacts["history"],
            df_movies=self.artifacts["movies"],
            df_users=self.artifacts["users"],
            query_prefix="inference",
        )

        X = features[metadata["feature_columns"]]

        pool = Pool(
            data=X,
            cat_features=metadata["cat_features"],
        )

        scored = features[["movie_id"]].copy()
        scored["score"] = ranker.predict(pool)

        top = scored.sort_values(["score", "movie_id"], ascending=[False, True]).head(k)

        return [
            {
                "movie_id": int(row.movie_id),
                "score": float(row.score),
            }
            for row in top.itertuples(index=False)
        ]
