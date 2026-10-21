import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import build_retrieval_data


def test_matrix_matches_original_ids(history):
    data = build_retrieval_data(history=history)

    assert data.user_item.shape == (3, 3)

    expected = np.zeros((3, 3), dtype=np.float32)

    expected[data.user_to_idx[20], data.movie_to_idx[500]] = 2
    expected[data.user_to_idx[100], data.movie_to_idx[900]] = 1
    expected[data.user_to_idx[7], data.movie_to_idx[300]] = 2

    np.testing.assert_array_equal(
        data.user_item.toarray(),
        expected,
    )

    for user_id, idx in data.user_to_idx.items():
        assert data.user_ids[idx] == user_id

    for movie_id, idx in data.movie_to_idx.items():
        assert data.movie_ids[idx] == movie_id


def test_negative_rating_seen_but_not_positive(history):
    data = build_retrieval_data(history=history)

    user_idx = data.user_to_idx[20]
    movie_idx = data.movie_to_idx[900]

    assert 900 in data.seen_movies[user_idx]
    assert data.user_item[user_idx, movie_idx] == 0


def test_build_retrieval_does_not_modify_history(history):
    original = history.copy()

    build_retrieval_data(history=history)

    pd.testing.assert_frame_equal(history, original)


def test_empty_history_is_rejected():
    with pytest.raises(ValueError, match="History must not be empty."):
        build_retrieval_data(history=pd.DataFrame())
