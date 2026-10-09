from dataclasses import dataclass

from scipy.sparse import csr_matrix
import numpy as np

@dataclass
class RetrievalData:
    """
    Класс данных для Retrieval

    Attributes:
        user_item (csr_matrix): Разряженная матрица взаимодействий
        movie_to_idx (dict[int, int]): Маппинг исходного movie_id -> movie_idx в ALS-матрице
        movie_ids (np.ndarray): Массив для восстановления индексов movie_idx -> movie_id
        seen_movies (dict[int, set[int]]): Словарь просмотренных фильмов
    """

    user_item: csr_matrix
    movie_to_idx: dict[int, int]
    movie_ids: np.ndarray
    seen_movies: dict[int, set[int]]
