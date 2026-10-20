import numpy as np
import pandas as pd


class Recommender:
    """
    Компонент рекомендательной система, основанный на популярности байеса.

    Модель собирает списки популярности и по формуле вычисляет
    IMDB-score, чем он выше тем релевантнее фильм

    После обучения модель может выдавать top-k рекомендаций для заданного пользователя,
    автоматически исключая просмотренные им фильмы

    Attributes:
        m (int): Вес общего среднего рейтинга при сглаживании.
        popularity (pd.DataFrame | None): Фильмы и их статистики,
            отсортированные по убыванию score после fit().
        user_watched (dict[int, set[int]]): Просмотренные фильмы
            для каждого внутреннего индекса пользователя.
        notes:
            Модель используется для разнообразия рекомендаций,
            а так же для возможности cold-start
    """

    def __init__(self, m: int = 100):
        self.m = m
        self.popularity: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame) -> "Recommender":
        """
        Обучает модель рекомендаций байеса

        На вход принимает матрицу взаимодействий `user-movie-rating`
        Считает среднее и кол-во оценок для `movie-rating`
        И вычисляет скор по формуле IMDB

        Args:
            interactions (pd.DataFrame): Матрица взаимодействий `user-movie-rating`

        Returns:
            Recommender: Обученную модель рекомендаций
        """

        C = interactions.rating.mean()

        popularity = (
            interactions.groupby("movie_id")
            .rating.agg(rating_count="count", avg_rating="mean")
            .reset_index()
        )

        popularity["score"] = (
            popularity.rating_count
            / (popularity.rating_count + self.m)
            * popularity.avg_rating
            + self.m / (popularity.rating_count + self.m) * C
        )

        self.popularity = popularity.sort_values("score", ascending=False).reset_index(
            drop=True
        )

        self.user_watched = (
            interactions.groupby("user_idx")["movie_id"].apply(set).to_dict()
        )

        return self

    def recommend(
        self,
        user_idx: int,
        exclude_movies: dict[int, set[int]] | None = None,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Возвращает top-k рекомендаций для заданного пользователя.

        Запрашивает ровно нужное кол-во пользователей, но не более кол-ва доступных фильмов
        Присутствует фильтрация:
        Уже просмотренные фильмы игнорируются

        Args:
            user_idx (int): Индекс пользователя
            exclude_movies (dict[int, set[int]] | None): Посмотренные пользователем фильмы
            k (int): Количество рекомендаций. По умолчанию `k` = 10

        Returns:
            list[tuple[int, float]]: Список пар `movie_id, score`. длина не превышает `k`.

        Raises:
            ValueError: Если количество рекомендаций отрицательно
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        if self.popularity.empty:
            raise ValueError("Popularity doesn't has items")

        watched_movies = (
            exclude_movies.get(user_idx, set())
            if exclude_movies is not None
            else self.user_watched.get(user_idx, set())
        )

        top = self.popularity.loc[~self.popularity.movie_id.isin(watched_movies)].head(
            k
        )

        return [
            (int(movie_id), float(score))
            for movie_id, score in zip(
                top["movie_id"].to_numpy(dtype=np.int64),
                top["score"].to_numpy(dtype=np.float64),
                strict=True,
            )
        ]
