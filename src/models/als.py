from implicit.als import AlternatingLeastSquares

from src.data.data import RetrievalData


class AlsRecommender:
    """
    Компонент рекомендательной системы, основанный на коллаборативной фильтрации
    с использованием ALS из библиотеки implicit

    Модель обучается на матрице взаимодействий "пользователь-фильм",
    построенной по положительной оценке (rating >= 4)
    в качестве веса взаимодействий используется (rating - 3)
    это усилит вклад положительных оценок при обучении

    После обучения модель может выдавать top-k рекомендаций для заданного пользователя,
    автоматически исключая просмотренные им фильмы

    Attributes:
        factors (int): Размерность латентных факторов пользователей и фильмов
        regularization (float): Коэффициент L2-регуляризации
        iterations (int): Количество итераций для обучения ALS
        random_state (int): Seed для воспроизводимости обучения
        alpha (float): Коэффициент уверенности
        num_threads (int): Количество используеммых потоков для обучения
        data (RetrievalData): Общие данные retrieval: матрица взаимодействий,
            соответствия индексов и просмотренные фильмы.
        model (AlternatingLeastSquares)
    """

    def __init__(
        self,
        factors: int = 16,
        regularization: float = 0.5,
        iterations: int = 10,
        random_state: int = 42,
        alpha: float = 1.0,
        num_threads: int = 6,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.alpha = alpha
        self.num_threads = num_threads

    def fit(self, data: RetrievalData) -> "AlsRecommender":
        """
        Модель обучает ALS на истории взаимодействимя пользователя с фильмами.

        На вход принимает собранный `RetrievalData` и обучает `AlternatingLeastSquares`
        Сам `RetrievalData` сохраняется в `self.data` и используется в `recommend`

        Args:
            data (RetrievalData): Контейнер, хранящий данные для Retireval

        Returns:
            AlsRecommender: self обученный экземпляр рекомендаций
        """
        self.data = data

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            num_threads=self.num_threads,
            random_state=self.random_state,
            alpha=self.alpha,
        )

        self.model.fit(self.data.user_item, show_progress=True)

        return self

    def recommend(self, user_idx: int, k: int = 500) -> list[tuple[int, float]]:
        """
        Возвращает top-k рекомендаций для заданного пользователя.

        Запрашивает у ALS больше кандидатов чем нужно (k + число просмотренных),
        что бы порсле фильтрации оставалось достаточно результатов.
        Фильтрация выполняется дважды:
        Со стороны модели ('filter_already_liked_items')
        и на стороне Python - по seen_movies

        Args:
            user_idx (int): Индекс пользователя
            k (int, optional): Кол-во рекомендуемых фильмов. По умолчанию `k` = 500

        Returns:
            list[tuple[int, float]]: Список пар `movie_id, score`. длина не превышает `k`.

        Raises:
            ValueError: Если количество рекомендаций отрицательно
        """

        if k <= 0:
            raise ValueError("k must be positive.")

        watched = self.data.seen_movies.get(user_idx, set())
        recommendations = []

        request_k = min(k + len(watched), self.data.user_item.shape[1])

        ids, scores = self.model.recommend(
            userid=int(user_idx),
            user_items=self.data.user_item[int(user_idx)],
            N=request_k,
            filter_already_liked_items=True,
        )

        for movie_idx, score in zip(ids, scores, strict=True):
            movie_id = self.data.movie_ids[int(movie_idx)]

            if movie_id in watched:
                continue

            recommendations.append((movie_id, float(score)))

            if len(recommendations) >= k:
                break

        return recommendations
