import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from src.data import RetrievalData

def build_retrieval_data(history: pd.DataFrame, n_users: int) -> RetrievalData:
    """
    Собирает данны для Retrieval

    Сохраняет уже просмотренные фильмы,
    так же делает маппинг идентификаторов пользователей
    и сохраняет массив по которому можно восстановить идентификаторы.
    Так же строит user-item матрицу для ALS и Similarity моделей

    Args:
        history (pd.DataFrame): История взаимодействий
        n_users (int): Количество пользователей
    Returns:
        RetrievalData: Контейнер для хранения данных
    """
    seen_movies = history.groupby("user_idx")["movie_id"].apply(set).to_dict()

    movie_ids = history["movie_id"].unique()
    
    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    positive = history.loc[
        history["rating"] >= 4,
        ["user_idx", "movie_id", "rating"]
    ].copy()

    positive["movie_idx"] = positive["movie_id"].map(movie_to_idx)
    values = (positive["rating"] - 3).values.astype(np.float32)

    user_item = csr_matrix(
        (
            values,
            (positive["user_idx"], positive["movie_idx"])
        ),
        shape=(n_users, len(movie_to_idx)),
        dtype=np.float32
    )

    return RetrievalData(
        user_item=user_item,
        movie_to_idx=movie_to_idx,
        movie_ids=movie_ids,
        seen_movies=seen_movies
    )
