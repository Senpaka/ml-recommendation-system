import numpy as np
import pandas as pd

def build_features(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    df_users: pd.DataFrame,
    df_movies: pd.DataFrame,
    query_prefix: str = "inference"
):
    
    data = candidates.copy()

    user_agg = (
        history.groupby("user_id")
        .agg(
            user_mean_rating=("rating", "mean"),
            user_rating_count=("rating", "count"),
            user_last_interaction=("timestamp", "max"),
            user_positive_rate=("rating", lambda x: (x >= 4).mean())
        )
        .reset_index()
    )

    movie_agg = (
        history.groupby("movie_id")
        .agg(
            movie_mean_rating=("rating", "mean"),
            movie_rating_count=("rating", "count"),
            movie_positive_rate=("rating", lambda x: (x >= 4).mean())
        )
        .reset_index()
    )

    data = data.merge(df_users, on="user_id", how="left")
    data = data.merge(
        df_movies.drop(columns=["title", "genres"], errors="ignore"),
        on="movie_id",
        how="left"
    )

    data = data.merge(user_agg, on="user_id", how="left")
    data = data.merge(movie_agg, on="movie_id", how="left")

    data["movie_mean_rating"] = data["movie_mean_rating"].fillna(history["rating"].mean())
    data["movie_rating_count"] = data["movie_rating_count"].fillna(0)
    data["movie_positive_rate"] = data["movie_positive_rate"].fillna(0)

    data["group_id"] = query_prefix + "_" + data["user_id"].astype(str)
    data = data.sort_values(["group_id", "movie_id"]).reset_index(drop=True)

    return data

def add_labels(data: pd.DataFrame, target: pd.DataFrame):
    target_labels = target[["user_id", "movie_id", "rating"]].copy()
    target_labels["label"] = np.where(
        target_labels["rating"] >= 4, 
        target_labels["rating"] - 2, 
        0
    )

    data = data.merge(
        target_labels[["user_id", "movie_id", "label"]],
        on=["user_id", "movie_id"],
        how="left"
    )

    data["label"] = data["label"].fillna(0).astype(np.int8)

    return data
