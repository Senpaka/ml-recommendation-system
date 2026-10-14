import numpy as np

from src.models.als import AlsRecommender
from src.models.recommender import Recommender
from src.models.similary import SimilarityRecommender
from src.generator import CandidateGenerator
from src.preprocessing import build_retrieval_data

import pandas as pd


class RetrievalPipeline:
    """
    Обучает retrieval-модели и формирует кандидатов для ранжирования.

    Подготавливает общие данные для ALS и Similarity, обучает три модели
    на одной истории взаимодействий и создаёт CandidateGenerator.
    Генератор объединяет рекомендации источников и отбирает кандидатов по RRF.

    Attributes:
        als_model (AlsRecommender): Модель рекомендаций на основе ALS.
        similarity_model (SimilarityRecommender): Модель сходства фильмов.
        popularity_model (Recommender): Модель рейтинга с байесовским сглаживанием.
        k_als (int): Максимальное количество кандидатов из ALS.
        k_sim (int): Максимальное количество кандидатов из Similarity.
        k_pop (int): Максимальное количество кандидатов из Popularity.
        candidate_k (int): Максимальное количество итоговых кандидатов
            на пользователя после отбора по RRF.
        generator (CandidateGenerator | None): Генератор кандидатов.
            Создаётся при успешном выполнении fit().
    """
    def __init__(
        self,
        als_params: dict,
        sim_params: dict,
        pop_params: dict,
        k_als: int = 500,
        k_sim: int = 1100,
        k_pop: int = 300,
        candidate_k: int = 500
    ):
        self.als_model = AlsRecommender(**als_params)
        self.similarity_model = SimilarityRecommender(**sim_params)
        self.popularity_model = Recommender(**pop_params)

        self.k_als = k_als
        self.k_sim = k_sim
        self.k_pop = k_pop
        self.candidate_k = candidate_k

        self.generator = None

    def fit(self, history: pd.DataFrame):
        """
        Обучает retrieval-модели на общей истории взаимодействий.

        Создаёт RetrievalData для ALS и Similarity, обучает модели
        и инициализирует генератор кандидатов.

        Args:
            history (pd.DataFrame): История взаимодействий с колонками
                movie_id, rating и timestamp.

        Returns:
            RetrievalPipeline: Обученный экземпляр пайплайна.
        """
        self.generator = None

        self.data = build_retrieval_data(history=history)

        indexed_history = history.copy()
        indexed_history["user_idx"] = (
            indexed_history["user_id"]
            .map(self.data.user_to_idx)
            .astype(np.intp)
        )

        self.als_model.fit(data=self.data)
        self.similarity_model.fit(data=self.data, history=indexed_history)
        self.popularity_model.fit(indexed_history)

        self.generator = CandidateGenerator(
            als_model=self.als_model,
            similarity_model=self.similarity_model,
            popularity_model=self.popularity_model,
            data=self.data,
            k_als=self.k_als,
            k_sim=self.k_sim,
            k_pop=self.k_pop,
            candidate_k=self.candidate_k
        )

        return self

    def make_candidates(self, users: pd.Series):
        """
        Формирует кандидатов для указанных пользователей.

        Объединяет рекомендации ALS, Similarity и Popularity,
        сохраняет признаки источников и оставляет до candidate_k
        фильмов на пользователя по убыванию RRF.

        Args:
            users (pd.Series): Уникальные идентификаторы пользователей
                (user_id).

        Returns:
            pd.DataFrame: Таблица кандидатов с user_idx, user_id, movie_id,
                скорами и рангами источников, признаками from_* и значением rrf.
                Если кандидатов нет, возвращает пустой DataFrame без колонок.

        Raises:
            RuntimeError: Если генератор ещё не создан вызовом fit().
            RuntimeError: Если в данных присутствует неизвестный индентификатор
        """
        if self.generator is None:
            raise RuntimeError("RetrievalPipeline must be fitted first.")

        user_ids = pd.Series(users).drop_duplicates()
        user_indices = user_ids.map(self.data.user_to_idx)

        unknown = user_ids.loc[user_indices.isna()]

        if not unknown.empty:
            raise ValueError(f"Unknown user IDs: {unknown.head(5).tolist()}")

        return self.generator.build_candidates(users=user_indices.to_numpy(dtype=np.intp))
        

