from dataclasses import dataclass

from scipy.sparse import csr_matrix
import numpy as np

@dataclass
class RetrievalData:
    """
    Класс данных для Retrieval

    Attributes:
        user_item (csr_matrix): Разряженная матрица взаимодействий
        movie_to_idx (dict[int, int]): Маппинг исходного movie_id -> movie_idx
        movie_ids (np.ndarray): Массив для восстановления индексов movie_idx -> movie_id
        user_to_idx (dict[int, int]): Маппинг исходного user_id -> user_idx
        user_ids: Массив для восстановления индексов user_idx -> user_id
        seen_movies (dict[int, set[int]]): Словарь просмотренных фильмов
    """

    user_item: csr_matrix

    user_to_idx: dict[int, int]
    user_ids: np.ndarray

    movie_to_idx: dict[int, int]
    movie_ids: np.ndarray

    seen_movies: dict[int, set[int]]
