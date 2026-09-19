import numpy as np
import pandas as pd

from src.models.recommender import Recommender
from src.models.matrix_factorization import FactorizationRecommender

from src.metrics.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k
)

def eval_pop(
    model: Recommender,
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    interactions: pd.DataFrame,
    k: int = 10
):

    recalls = []
    precisions = []
    ndcgs = []

    for user_id in interactions.user_id.unique():

        recommended = model.recommend(
            user_id=user_id, 
            interactions=train, 
            k=k
        )

        val_user = val[val.user_id == user_id]

        relevant = val_user.loc[
            val_user["rating_binary"] == 1,
            "movie_id"
        ].tolist()

        if len(relevant) == 0:
            continue

        recalls.append(recall_at_k(relevant, recommended, k))
        precisions.append(precision_at_k(relevant, recommended, k))
        ndcgs.append(ndcg_at_k(relevant, recommended, k))

    return {
        f"recall@{k}": np.mean(recalls),
        f"precision@{k}": np.mean(precisions),
        f"NDCG@{k}": np.mean(ndcgs)
    }

def eval_factorization(
    model: FactorizationRecommender,
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    k=10
):

    recalls = []
    precisions = []
    ndcgs = []

    for user_idx in val.user_idx.unique():

        recommendations = model.recommend(train, user_idx, k)
        val_user = val[val.user_idx == user_idx]

        recommended = [
            movie_idx
            for movie_idx, _ in recommendations
        ]

        relevant = val_user.loc[
            val_user.rating >= 4,
            "movie_idx"
        ].tolist()

        if len(relevant) == 0:
            continue

        recalls.append(recall_at_k(relevant, recommended, k))
        precisions.append(precision_at_k(relevant, recommended, k))
        ndcgs.append(ndcg_at_k(relevant, recommended, k))

    return {
        f"recall@{k}": np.mean(recalls),
        f"precision@{k}": np.mean(precisions),
        f"NDCG@{k}": np.mean(ndcgs)
    }
