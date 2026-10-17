import numpy as np


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    y_pred = y_pred[:k]

    hits = len(set(y_true) & set(y_pred))

    return hits / k


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    y_pred = y_pred[:k]

    hits = len(set(y_pred) & set(y_true))

    return hits / len(y_true) if len(y_true) > 0 else 0.0


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    relevant = set(y_true)
    predictions = list(y_pred)[:k]

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, movie_id in enumerate(predictions)
        if movie_id in relevant
    )

    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))

    return float(dcg / idcg) if idcg > 0 else 0.0
