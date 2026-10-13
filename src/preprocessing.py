import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from src.data import RetrievalData

MAX_VALUE_ID = 2_147_483_647

# =========================Основной метод препроцессинга=========================

def preprocess_data(
    raw_users: pd.DataFrame,
    raw_movies: pd.DataFrame,
    raw_interactions: pd.DataFrame
):
    return(
        preprocess_users(raw_users),
        preprocess_movies(raw_movies),
        preprocess_interactions(raw_interactions)
    )

# =========================методы препроцессинга=========================

def preprocess_movies(df: pd.DataFrame):
    df = df.copy()

    df = _validate_id(df, "movie_id")
    df = _movie_handle_missing(df)

    df = df.astype({
        "movie_id": "int32", 
        "title": "string",
        "genre": "string"
    })

    df["year"] = pd.to_numeric(
        df["title"].str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    )

    genres = df.genre.str.get_dummies('|')

    return pd.concat([df, genres], axis=1)


def preprocess_users(df: pd.DataFrame):
    df = df.copy()

    df = _user_validate_data(df)
    df = _user_handle_missing(df)

    df['gender'] = (
        df['gender']
        .str.strip()
        .str.upper()
        .map({'F': 0, 'M': 1})
        .fillna(-1)
    )

    return df.astype({
        "user_id": "int32",
        "gender": "int8",
        "age_group": "int8",
        "occupation": "int8",
        "zip_code": "string",
    })


def preprocess_interactions(df: pd.DataFrame):
    df = df.copy()

    df = _interactions_validate_data(df)
    df = _interactions_handle_missing(df)

    return df.astype({
        "user_id": "int32",
        "movie_id": "int32",
        "rating": "float32",
        "timestamp": "int64"
    })

# =========================Вспомогательные методы для фильмов=========================

def _movie_handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["movie_id"])

    df["title"] = df["title"].fillna("unknown")
    df["genre"] = df["genre"].fillna("unknown")

    return df

# =========================Вспомогательные методы для пользователей=========================

def _user_handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset="user_id")

    df["age_group"] = df["age_group"].fillna(-1)
    df["occupation"] = df["occupation"].fillna(-1)
    df["zip_code"] = df["zip_code"].fillna("unknown")

    return df


def _user_validate_data(df: pd.DataFrame):
    df = _validate_id(df, "user_id")

    allowed_values = {
        "age_group": {1, 18, 25, 35, 45, 50, 56, 70},
        "occupation": set(range(21))
    }

    for column, allowed in allowed_values.items():
        values = pd.to_numeric(df[column], errors="coerce")
        df[column] = values.where(values.isin(allowed)) 

    return df

# =========================Вспомогательные методы для взаимодействий=========================

def _interactions_validate_data(df: pd.DataFrame):
    df = _validate_id(df, "user_id")
    df = _validate_id(df, "movie_id")

    ratings = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = ratings.where(ratings.isin([1, 2, 3, 4, 5]))

    timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    valid_timestamp = (
        timestamp.notna()
        & np.isfinite(timestamp)
        & timestamp.ge(0)
        & timestamp.lt(2**63)
        & timestamp.mod(1).eq(0)
    )
    df["timestamp"] = timestamp.where(valid_timestamp)

    return df

def _interactions_handle_missing(df: pd.DataFrame):
    df = df.dropna(subset=["user_id", "movie_id", "rating", "timestamp"]).copy()

    return df

# =========================Общие вспомогательные методы=========================

def _validate_id(df: pd.DataFrame, name_id: str):
    ids = pd.to_numeric(df[name_id], errors="coerce")
    
    valid = (
        ids.notna()
        & ids.between(1, MAX_VALUE_ID)
        & ids.mod(1).eq(0)
    )

    df[name_id] = ids.where(valid)

    return df

# =========================Методы для Retrieval=========================

def build_retrieval_data(history: pd.DataFrame) -> RetrievalData:
    """
    Собирает данны для Retrieval

    Сохраняет уже просмотренные фильмы,
    так же делает маппинг идентификаторов фильмов
    и сохраняет массив по которому можно восстановить идентификаторы.
    Так же строит user-item матрицу для ALS и Similarity моделей

    Args:
        history (pd.DataFrame): История взаимодействий
        
    Returns:
        RetrievalData: Контейнер для хранения данных
    
    Raises:
    ValueError: Если передана пустая история
    """

    if history.empty:
        raise ValueError("History must not be empty.")

    history = history.copy()

    user_ids = history["user_id"].unique()
    movie_ids = history["movie_id"].unique()

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }
    movie_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }

    history["user_idx"] = history["user_id"].map(user_to_idx)
    history["movie_idx"] = history["movie_id"].map(movie_to_idx)

    seen_movies = history.groupby("user_idx")["movie_id"].apply(set).to_dict()

    positive = history.loc[
        history["rating"] >= 4,
    ].copy()

    user_item = csr_matrix(
        (
            (positive["rating"] - 3).to_numpy(dtype=np.float32),
            (
                positive["user_idx"].to_numpy(dtype=np.intp), 
                positive["movie_idx"].to_numpy(dtype=np.intp)
            )
        ),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32
    )

    return RetrievalData(
        user_item=user_item,
        movie_to_idx=movie_to_idx,
        movie_ids=movie_ids,
        seen_movies=seen_movies,
        user_to_idx=user_to_idx,
        user_ids=user_ids
    )