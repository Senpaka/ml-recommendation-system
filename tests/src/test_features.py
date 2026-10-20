import pandas as pd
import pytest

from src.features.features import add_labels, build_features


def test_aggregates_and_unknown_movie_fallback_use_history_only(history):

    candidates = pd.DataFrame({"user_id": [20, 100], "movie_id": [900, 999]})

    users = pd.DataFrame({"user_id": [20, 100], "occupation": [3, 7]})

    movies = pd.DataFrame(
        {"movie_id": [900, 999], "title": ["A", "B"], "year": [2000, 2001]}
    )

    result = build_features(
        candidates=candidates,
        history=history,
        df_movies=movies,
        df_users=users,
        query_prefix="test",
    ).set_index("user_id")

    assert len(result) == len(candidates)
    assert result.loc[20, "user_mean_rating"] == 3.5
    assert result.loc[20, "user_rating_count"] == 2
    assert result.loc[20, "user_positive_rate"] == 0.5
    assert result.loc[20, "user_last_interaction"] == 3
    assert result.loc[20, "movie_mean_rating"] == 3.0
    assert result.loc[20, "movie_rating_count"] == 2
    assert result.loc[100, "movie_mean_rating"] == pytest.approx(4.0)
    assert result.loc[100, "movie_rating_count"] == 0
    assert result.loc[100, "movie_positive_rate"] == 0
    assert result.loc[20, "group_id"] == "test_20"
    assert "title" not in result
    assert "label" not in result


def test_labels_join_on_user_and_movie_without_adding_candidates():
    candidates = pd.DataFrame(
        {"user_id": [20, 20, 100, 100], "movie_id": [500, 900, 500, 300]}
    )

    target = pd.DataFrame(
        {
            "user_id": [20, 20, 100, 7],
            "movie_id": [500, 900, 500, 999],
            "rating": [5, 4, 2, 5],
        }
    )

    before = candidates.copy()

    result = add_labels(data=candidates, target=target)

    assert result["label"].tolist() == [3, 2, 0, 0]
    assert len(result) == len(candidates)

    pd.testing.assert_frame_equal(result.drop(columns="label"), before)
