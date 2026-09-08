
import numpy as np

def precision_at_k(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
) -> float:

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    if k > len(y_pred):
        raise ValueError("k must be less than or equal to the length of y_pred.")
    
    y_pred = y_pred[:k]

    hits = len(
        set(y_true) & set(y_pred)
    )

    return hits / k 

def recall_at_k(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
) -> float:

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if k > len(y_pred):
        raise ValueError("k must be less than or equal to the length of y_pred.")

    y_pred = y_pred[:k]

    hits = len(
        set(y_pred) & set(y_true)
    )

    return hits / len(y_true) if len(y_true) > 0 else 0.0

def ndcg_at_k(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int = 10
) -> float:

    if k <= 0:
            raise ValueError("k must be a positive integer.")
    
    if k > len(y_pred):
        raise ValueError("k must be less than or equal to the length of y_pred.")

    y_pred = y_pred[:k]

    dcg = 0.0

    for i, item in enumerate(y_pred):
        if item in y_true:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(
        1 / np.log2(i + 2) for i in range(min(len(y_true), k))
    )

    return dcg / idcg if idcg > 0 else 0.0