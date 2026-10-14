from src.data.data import RetrievalData
from src.models.als import AlsRecommender
from src.models.recommender import Recommender
from src.models.similary import SimilarityRecommender

import pandas as pd
import numpy as np

class CandidateGenerator:
    """
    Генератор построения кандидатов для ранжирования

    Генератор берет рекомендации для каждого пользователя
    Строит для них ранк и rrf-score
    
    После прохода по всем пользователям кандидаты собираются в единый `pd.DataFrame`
    и сортируются по rrf-score 

    Attributes:
        k_als (int): Количество кандидатов из ALS модели
        k_sim (int): Количество кандидатов из Similarity модели
        k_pop (int): Количество кандидатов из IMDB модели
        candidate_k (int): Лимит кандидатов на 1 пользователя
        data RetrievalData: Контейнер для хранения данных
        als_model (AlsRecommender): Обученная ALS модель
        similarity_model (SimilarityRecommender): Обученная Similarity модель
        popularity_model (Recommender): Обученная IMDB модель
    """
    def __init__(
        self,
        als_model: AlsRecommender,
        similarity_model: SimilarityRecommender,
        popularity_model: Recommender,
        data: RetrievalData,
        k_als: int = 500,
        k_sim: int = 1100,
        k_pop: int = 300,
        candidate_k: int = 500
    ):
        self.data = data
        self.k_als = k_als
        self.k_sim = k_sim
        self.k_pop = k_pop
        self.candidate_k = candidate_k
        self.als_model = als_model
        self.similarity_model = similarity_model
        self.popularity_model = popularity_model

    def build_candidates(self, users: np.ndarray) -> pd.DataFrame:
        """
        Построение кандидатов

        Проходимся по всем пользователям,
        забираем рекомендации с каждой модели,
        добавляем ранг фильма и считаем rrf
        и добавляем в `rows`.
        Если рекомендаций не окажется то возвращаем пустой `pd.DataFrame`

        Args:
            users (np.ndarray): Массив уникальных идентификаторов пользователей 
            
        Returns:
            pd.DataFrame: Кандидаты, полученные из трех моделей
        """
        rows = []

        for user_idx in users:
            candidates = {}

            als_candidates = self.als_model.recommend(user_idx, k=self.k_als)
            sim_candidates = self.similarity_model.recommend(user_idx, k=self.k_sim)
            pop_candidates = self.popularity_model.recommend(user_idx, k=self.k_pop)

            self._merge_source_candidates(candidates, als_candidates, "als")
            self._merge_source_candidates(candidates, sim_candidates, "sim")
            self._merge_source_candidates(candidates, pop_candidates, "pop")          
            self._append_candidate_rows(user_idx, rows, candidates)
            
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

        result["user_id"] = self.data.user_ids[
            result["user_idx"].to_numpy(dtype=np.intp)
        ]

        return result

    @staticmethod
    def _merge_source_candidates(candidates: dict, model_candidates: list[tuple[int, float]], model_name: str) -> None:
        """
        Присваивание ранга фильму для любой модели

        проходимся по всем фильмам и присваиваем им ранг (1,...)
        так же укажем что фильм рекомендуется из этой модели

        Args:
            candidates (dict): Текущие кандидаты
            model_candidates (list[tuple[int, float]]): Кандидаты, рекомендованные моделью
            model_name (str): Название модели
        """
        for rank, (movie_id, score) in enumerate(model_candidates, start=1):
            candidates.setdefault(movie_id, {})
            candidates[movie_id].update({
                f"{model_name}_score": float(score),
                f"{model_name}_rank": rank,
                f"from_{model_name}": 1
            })

    @staticmethod
    def _append_candidate_rows(user_idx: int, rows: list[dict], candidates: dict) -> None:
        """
        Добавляет rrf к каждому фильму

        проходится по всем кандидатам и для каждого считает rff

        Args:
            user_idx (int): Индекс пользователя
            rows (list[dict]): Строки для построения кандидатов
            candidates (dict): Текущие кандидаты из трех моделей
        """
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
