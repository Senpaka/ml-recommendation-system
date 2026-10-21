import pytest

from src.models.recommender import Recommender


def test_recommendations_exclude_seen_movies(popularity_history):

    model = Recommender()
    model.fit(popularity_history)

    recommendations = model.recommend(user_idx=0, k=10)

    movie_ids = [movie_id for movie_id, _ in recommendations]

    assert set(movie_ids) == {300, 400}
    assert len(set(movie_ids)) == len(movie_ids)

    limited = model.recommend(user_idx=0, k=1)

    assert len(limited) == 1
    assert limited == recommendations[:1]


def test_recommend_raises(popularity_history, small_history):
    model = Recommender()
    model.fit(popularity_history)

    with pytest.raises(ValueError, match="k must be positive."):
        model.recommend(user_idx=0, k=0)

    model = Recommender()
    model.fit(small_history)

    with pytest.raises(ValueError, match="Popularity recommendations are unavailable."):
        model.recommend(user_idx=0, k=10)
