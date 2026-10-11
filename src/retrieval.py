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
        user_encoder: Энкодер для преобразования user_id и user_idx.
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
        user_encoder,
        k_als: int = 500,
        k_sim: int = 1100,
        k_pop: int = 300,
        candidate_k: int = 500
    ):
        self.als_model = AlsRecommender(**als_params)
        self.similarity_model = SimilarityRecommender(**sim_params)
        self.popularity_model = Recommender(**pop_params)

        self.user_encoder = user_encoder

        self.k_als = k_als
        self.k_sim = k_sim
        self.k_pop = k_pop
        self.candidate_k = candidate_k

        self.generator = None

    def fit(self, history: pd.DataFrame, n_users: int):
        """
        Обучает retrieval-модели на общей истории взаимодействий.

        Создаёт RetrievalData для ALS и Similarity, обучает модели
        и инициализирует генератор кандидатов.

        Args:
            history (pd.DataFrame): История взаимодействий с колонками
                user_idx, movie_id, rating и timestamp.
                user_idx должен соответствовать переданному user_encoder.
            n_users (int): Количество строк пользовательской матрицы.
                Все user_idx должны находиться в диапазоне [0, n_users).

        Returns:
            RetrievalPipeline: Обученный экземпляр пайплайна.
        """
        data = build_retrieval_data(history=history, n_users=n_users)

        self.als_model.fit(data=data)
        self.similarity_model.fit(data=data, history=history)
        self.popularity_model.fit(history)

        self.generator = CandidateGenerator(
            als_model=self.als_model,
            similarity_model=self.similarity_model,
            popularity_model=self.popularity_model,
            user_encoder=self.user_encoder,
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
            users (pd.Series): Уникальные внутренние индексы пользователей
                (user_idx).

        Returns:
            pd.DataFrame: Таблица кандидатов с user_idx, user_id, movie_id,
                скорами и рангами источников, признаками from_* и значением rrf.
                Если кандидатов нет, возвращает пустой DataFrame без колонок.

        Raises:
            RuntimeError: Если генератор ещё не создан вызовом fit().
        """
        if self.generator is None:
            raise RuntimeError("RetrievalPipeline must be fitted first.")

        return self.generator.build_candidates(users=users)
        

