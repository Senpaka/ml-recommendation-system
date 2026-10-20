import numpy as np
import pandas as pd
import pytest 

from src.data.preprocessing import (
    preprocess_interactions,
    preprocess_movies,
    preprocess_users
)


def test_movies_handle_missing_values_and_extract_features():
    raw = pd.DataFrame(
        {
            "movie_id": ["500", "900", "bad"],
            "title": ["Film (1999)", None, "Discarded"],
            "genre": ["Action|Comedy", None, "Drama"],
        }
    )

    original = raw.copy()
    result = preprocess_movies(raw).set_index("movie_id")

    assert set(result.index) == {500, 900}
    assert result.loc[500, "year"] == 1999
    assert result.loc[500, "Action"] == result.loc[500, "Comedy"] == 1
    assert result.loc[900, "title"] == "unknown"
    assert result.loc[900, "genre"] == "unknown"
    assert pd.isna(result.loc[900, "year"])
    pd.testing.assert_frame_equal(raw, original)


def test_users_normalize_categories_without_losing_zip_zeroes():
    raw = pd.DataFrame(
        {
            "user_id": [20, 100, 0],
            "gender": [" f ", "?", "M"],
            "age_group": ["25", "999", 18],
            "occupation": ["3", "bad", 1],
            "zip_code": ["00123", None, "99999"],
        }
    )

    original = raw.copy()
    result = preprocess_users(raw).set_index("user_id")

    assert set(result.index) == {20, 100}
    assert result.loc[20, "gender"] == 0
    assert result.loc[20, "age_group"] == 25
    assert result.loc[20, "zip_code"] == "00123"
    assert result.loc[100, ["gender", "age_group", "occupation"]].tolist() == [-1, -1, -1]
    assert result.loc[100, "zip_code"] == "unknown"
    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize(
    "column, value",
    [
        ("user_id", 0),
        ("user_id", 1.5),
        ("user_id", 2**31),
        ("movie_id", "bad"),
        ("movie_id", -1),
        ("rating", 0),
        ("rating", 6),
        ("rating", 4.5),
        ("timestamp", -1),
        ("timestamp", np.inf),
        ("timestamp", 1.5),
        ("timestamp", "bad"),
    ],
)
def test_invalid_interaction_is_removed(column, value):
    good = {"user_id": "20", "movie_id": "500", "rating": "5", "timestamp": "42"}
    raw = pd.DataFrame([good, {**good, column: value}])
    original = raw.copy()
    result = preprocess_interactions(raw)

    assert len(result) == 1
    assert result.iloc[0].to_dict() == {
        "user_id": 20,
        "movie_id": 500,
        "rating": 5,
        "timestamp": 42,
    }
    assert str(result["rating"].dtype) == "float32"
    assert str(result["timestamp"].dtype) == "int64"
    pd.testing.assert_frame_equal(raw, original)