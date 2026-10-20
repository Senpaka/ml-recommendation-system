import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import build_retrieval_data
from src.models.similary import SimilarityRecommender


def test_similarity_finds_co_rated_unseen_movie():
    history = pd.DataFrame(
        {
            "user_id": [20, 100, 100, 7],
            "movie_id": [500, 500, 900, 300],
            "rating": [5, 5, 5, 5],
            "timestamp": [1, 1, 2, 1],
        }
    )

    data = build_retrieval_data(history=history)
    history["user_idx"] = history.user_id.map(data.user_to_idx)

    model = SimilarityRecommender(k_neighbors=2).fit(data=data, history=history)
    results = model.recommend(data.user_to_idx[20], k=1)

    assert len(results) == 1
    assert results[0][0] == 900
    assert results[0][1] == pytest.approx(1 / np.sqrt(2))

    for idx, (neighbors, _) in model.neighbors.items():
        assert idx not in neighbors


def test_recent_seeds_and_users_without_positive_ratings():
    history = pd.DataFrame(
        {
            "user_id": [20, 20, 20, 100],
            "movie_id": [500, 900, 300, 500],
            "rating": [5, 4, 2, 1],
            "timestamp": [30, 10, 40, 20],
        }
    )

    data = build_retrieval_data(history=history)
    history["user_idx"] = history.user_id.map(data.user_to_idx)

    model = SimilarityRecommender(max_sim_seeds=1).fit(data, history)

    assert model.liked_recent[data.user_to_idx[20]] == [500]
    assert model.recommend(data.user_to_idx[100], k=10) == []


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError):
        SimilarityRecommender().recommend(0, k=k)