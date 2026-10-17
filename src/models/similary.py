import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.data.data import RetrievalData


class SimilarityRecommender:
    """
    Компонент рекомендательной системы Item2Item, основанный на методе ближайших соседей
    Использует `sklearn.neighbors.NearestNeighbors` для поиска релевантности

    Модель обучается на матрице взаимодействий "пользователь-фильм",
    построенной по положительной оценке (rating >= 4)
    в качестве веса взаимодействий используется (rating - 3)
    это усилит вклад положительных оценок при обучении

    После обучения модель может выдавать top-k рекомендаций для заданного пользователя,
    автоматически исключая просмотренные им фильмы

    Attributes:
        k_neighbors (int): Количество соседей для поиска
        max_sim_seeds (int): Максимальное число последних положительно оценённых фильмов, используемых для поиска похожих
        movie_user (csr_matrix): Транспонированная и нормализованная по строкам матрица
        model (NearestNeighbors): Модель поиска ближайших соседей
        neighbors (dict[int: tuple]): Словарь с наиболее похожими фильмами.
            Внутренний индекс фильма → пара массивов индексов соседей и сходств
        liked_recent (dict[int, list[int]]): Фильмы, которые пользователь оценил положительно
        data (RetrievalData): Общие данные retrieval: матрица взаимодействий,
            соответствия индексов и просмотренные фильмы.
    """

    def __init__(
        self,
        k_neighbors: int = 100,
        max_sim_seeds: int = 50,
    ):
        self.k_neighbors = k_neighbors
        self.max_sim_seeds = max_sim_seeds

    def fit(
        self, data: RetrievalData, history: pd.DataFrame
    ) -> "SimilarityRecommender":
        """
        Модель обучает `NearestNeighbors` на истории взаимодействий фильмов с пользователями

        На вход принимает собранный `RetrievalData` и обучает `NearestNeighbors`
        Сам `RetrievalData` сохраняется в `self.data` и используется в `recommend`

        args:
            data (RetrievalData): Контейнер, хранящий данные для Retireval
            history (pd.DataFrame): История взаимодействий

        return:
            SimilarityRecommender: self обученный экземпляр рекомендаций
        """

        self.data = data

        self.movie_user = normalize(self.data.user_item.T.tocsr(), axis=1)

        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        self.model.fit(self.movie_user)

        n_neighbors = min(self.k_neighbors + 1, self.movie_user.shape[0])

        distances, indices = self.model.kneighbors(
            self.movie_user,
            n_neighbors=n_neighbors,
        )

        self.neighbors = {}

        for movie_idx in range(self.movie_user.shape[0]):
            mask = indices[movie_idx] != movie_idx

            self.neighbors[movie_idx] = (
                indices[movie_idx][mask][: self.k_neighbors],
                (1.0 - distances[movie_idx][mask])[: self.k_neighbors],
            )

        self.liked_recent = (
            history.loc[history["rating"] >= 4]
            .sort_values(["user_idx", "timestamp"])
            .groupby("user_idx")
            .tail(self.max_sim_seeds)
            .groupby("user_idx")["movie_id"]
            .apply(list)
            .to_dict()
        )

        return self

    def recommend(self, user_idx: int, k: int = 500) -> list[tuple[int, float]]:
        """
        Возвращает до k непросмотренных фильмов по сходству с историей пользователя.

        Использует последние max_sim_seeds положительно оценённых фильмов.
        Для каждого кандидата выбирает максимальное сходство среди его
        совпадений с этими фильмами и сортирует кандидатов по убыванию скора.

        Args:
            user_idx (int): Внутренний индекс пользователя.
            k (int): Максимальное количество рекомендаций.

        Returns:
            list[tuple[int, float]]: Пары (movie_id, score).

        Raises:
            ValueError: Если количество рекомендаций отрицательно
        """
        if k <= 0:
            raise ValueError("k must be positive.")

        watched = self.data.seen_movies.get(user_idx, set())

        sim_scores: dict[int, float] = {}

        for seed_movie_id in self.liked_recent.get(user_idx, []):
            seed_idx = self.data.movie_to_idx.get(seed_movie_id)

            if seed_idx is None:
                continue

            neighbor_ids, neighbor_scores = self.neighbors[seed_idx]

            for neighbor_idx, score in zip(neighbor_ids, neighbor_scores, strict=True):
                movie_id = self.data.movie_ids[neighbor_idx]

                if movie_id in watched:
                    continue

                sim_scores[movie_id] = max(sim_scores.get(movie_id, 0), score)

        sim_top = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return sim_top
