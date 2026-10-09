from src.models.als import AlsRecommender
from src.models.recommender import Recommender
from src.models.similary import SimilarityRecommender

import pandas as pd

class CandidateGenerator:

    def __init__(
        self,
        als_model: AlsRecommender,
        similarity_model: SimilarityRecommender,
        popularity_model: Recommender,
        user_encoder,
        k_als: int = 500,
        k_sim: int = 1100,
        k_pop: int = 300,
        candidate_k: int = 500
    ):

        self.k_als = k_als
        self.k_sim = k_sim
        self.k_pop = k_pop
        self.candidate_k = candidate_k
        self.user_encoder = user_encoder
        self.als_model = als_model
        self.similarity_model = similarity_model
        self.popularity_model = popularity_model

    def build_candidates(self, users):

        rows = []

        for user_idx in users:
            candidates = {}

            als_candidates = self.als_model.recommend(user_idx, k=self.k_als)

            sim_candidates = self.similarity_model.recommend(user_idx, k=self.k_sim)

            pop_candidates = self.popularity_model.recommend(user_idx, k=self.k_pop)

            self._build_rank(als_candidates, "als")

            self._build_rank(sim_candidates, "sim")

            self._build_rank(pop_candidates, "pop")          

            self._build_row(user_idx, rows, candidates)
            

        result = pd.DataFrame(rows)

        if result.empty:
            return result

        result = (
            result.sort_values(
                ["user_idx", "rrf"],
                ascending=[True, False]
            )
            .groupby("user_idx")
            .head(self.candidate_k)
            .reset_index(drop=True)
        )

        result["user_id"] = self.user_encoder.inverse_transform(result["user_idx"])

        return result

    @staticmethod
    def _build_rank(candidates: dict, model_candidates: list[int, float], model_name: str, ):
        for rank, (movie_id, score) in enumerate(model_candidates, start=1):
            candidates.setdefault(movie_id, {})
            candidates[movie_id].update({
                f"{model_name}_score": float(score),
                f"{model_name}_rank": rank,
                f"from_{model_name}": 1
            })

    @staticmethod
    def _build_row(user_idx: int, rows: list[dict], candidates: dict):

        for movie_id, features in candidates.items():
        
            rrf = 0.

            if features.get("from_als"):
                rrf += 1 / (60 + features["als_rank"])

            if features.get("from_sim"):
                rrf += 1 / (60 + features["sim_rank"])

            if features.get("from_pop"):
                rrf += 1 / (60 + features["pop_rank"])

            rows.append({
                "user_idx": int(user_idx),
                "movie_id": int(movie_id),

                "als_score": features.get("als_score", 0.0),
                "sim_score": features.get("sim_score", 0.0),
                "pop_score": features.get("pop_score", 0.0),

                "als_rank": features.get("als_rank", 0),
                "sim_rank": features.get("sim_rank", 0),
                "pop_rank": features.get("pop_rank", 0),

                "from_als": features.get("from_als", 0),
                "from_sim": features.get("from_sim", 0),
                "from_pop": features.get("from_pop", 0),

                "rrf": rrf
            })
