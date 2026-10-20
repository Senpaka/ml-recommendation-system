from src.config import Config
from src.models.als import AlsRecommender
from src.retrieval import build_retrieval_data

import numpy as np
import pytest


def test_recommendations_exclude_seen_movies(history):
    data = build_retrieval_data(history)

    model = AlsRecommender(factors=2, iterations=2, num_threads=1)
    model.fit(data)

    user_idx = data.user_to_idx[20]

    recommendations = model.recommend(user_idx=user_idx, k=10)

    assert [movie for movie, _ in recommendations] == [300]
    assert all(np.isfinite(score) for _, score in recommendations)
    assert model.recommend(user_idx=user_idx, k=1) == recommendations[:1]


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must be positive"):
        AlsRecommender().recommend(user_idx=0, k=k)


def test_all_movies_seen_returns_empty(history):
    history = history.copy()
    history["user_id"] = 20

    data = build_retrieval_data(history=history)

    model = AlsRecommender(factors=2, iterations=2, num_threads=1).fit(data)

    assert model.recommend(data.user_to_idx[20], k=10) == []