from unittest.mock import MagicMock, Mock, sentinel

import pandas as pd
import pytest

import services.recommendations_service.recommendation as service_module


@pytest.fixture
def dependencies(monkeypatch):
    retrieval = MagicMock()
    retrieval.can_personalize.return_value = True
    retrieval.make_candidates.return_value = pd.DataFrame(
        {
            "user_id": [42, 42, 42],
            "movie_id": [30, 20, 10],
        }
    )

    ranker = Mock()
    ranker.predict.return_value = [0.4, 0.9, 0.9]

    features = pd.DataFrame(
        {
            "user_id": [42, 42, 42],
            "movie_id": [30, 20, 10],
            "feature_a": [3.0, 2.0, 1.0],
            "unused_feature": [100, 200, 300],
        }
    )

    build_features = Mock(return_value=features)
    pool = Mock(return_value=sentinel.pool)

    monkeypatch.setattr(service_module, "build_features", build_features)
    monkeypatch.setattr(service_module, "Pool", pool)

    return {
        "retrieval": retrieval,
        "ranker": ranker,
        "build_features": build_features,
        "pool": pool,
    }


@pytest.fixture
def service(dependencies):
    artifacts = {
        "retrieval": dependencies["retrieval"],
        "ranker": dependencies["ranker"],
        "metadata": {
            "feature_columns": ["feature_a"],
            "cat_features": [],
        },
        "history": pd.DataFrame(),
        "movies": pd.DataFrame(),
        "users": pd.DataFrame(),
    }

    return service_module.RecommendationService(artifacts)


@pytest.mark.parametrize("k", [0, -1])
def test_recommend_rejects_non_positive_k(service, dependencies, k):
    with pytest.raises(ValueError, match="k must be positive"):
        service.recommend(user_id=42, k=k)

    dependencies["retrieval"].can_personalize.assert_not_called()
    dependencies["ranker"].predict.assert_not_called()


def test_cold_start_returns_popular_without_ranking(service, dependencies):
    retrieval = dependencies["retrieval"]

    retrieval.can_personalize.return_value = False
    retrieval.recommend_popular.return_value = [
        (100, 4.8),
        (200, 4.6),
    ]

    result = service.recommend(user_id=999, k=2)

    assert result == [
        {"movie_id": 100, "score": 4.8},
        {"movie_id": 200, "score": 4.6},
    ]

    retrieval.can_personalize.assert_called_once_with(999)
    retrieval.recommend_popular.assert_called_once_with(
        user_id=999,
        k=2,
    )

    retrieval.make_candidates.assert_not_called()

    dependencies["build_features"].assert_not_called()
    dependencies["pool"].assert_not_called()
    dependencies["ranker"].assert_not_called()


def test_personalized_recommender_are_sorted_and_limited(service, dependencies):
    result = service.recommend(user_id=42, k=2)

    assert result == [
        {"movie_id": 10, "score": 0.9},
        {"movie_id": 20, "score": 0.9},
    ]

    retrieval = dependencies["retrieval"]

    retrieval.can_personalize.assert_called_once_with(42)
    retrieval.make_candidates.assert_called_once_with(users=[42])
    retrieval.recommend_popular.assert_not_called()

    dependencies["ranker"].predict.assert_called_once_with(sentinel.pool)


def test_feature_receive_artifacts_and_pool_uses_metadata(service, dependencies):
    service.recommend(user_id=42)

    build_features = dependencies["build_features"]
    build_features.assert_called_once()
    kwargs = build_features.call_args.kwargs

    assert kwargs["history"] is service.artifacts["history"]
    assert kwargs["df_movies"] is service.artifacts["movies"]
    assert kwargs["df_users"] is service.artifacts["users"]
    assert kwargs["query_prefix"] == "inference"

    pool = dependencies["pool"]
    pool.assert_called_once()

    expected = build_features.return_value[["feature_a"]]

    pd.testing.assert_frame_equal(
        pool.call_args.kwargs["data"],
        expected,
    )

    assert pool.call_args.kwargs["cat_features"] == []


def test_empty_candidates_return_empty_without_ranking(service, dependencies):
    dependencies["retrieval"].make_candidates.return_value = pd.DataFrame()

    result = service.recommend(user_id=42)

    assert result == []
    dependencies["build_features"].assert_not_called()
    dependencies["pool"].assert_not_called()
    dependencies["ranker"].predict.assert_not_called()


def test_return_avaible_candidates_when_fewer_than_k(service):
    result = service.recommend(user_id=42, k=10)

    assert result == [
        {"movie_id": 10, "score": 0.9},
        {"movie_id": 20, "score": 0.9},
        {"movie_id": 30, "score": 0.4},
    ]


def test_empty_popularity_returns_empty(service, dependencies):
    retrieval = dependencies["retrieval"]
    retrieval.can_personalize.return_value = False
    retrieval.recommend_popular.return_value = []

    assert service.recommend(user_id=999) == []
    dependencies["ranker"].predict.assert_not_called()


@pytest.mark.parametrize("as_string", [False, True])
def test_from_artifacts_loads_existing_directory(tmp_path, monkeypatch, as_string):
    loader = Mock(return_value=sentinel.artifacts)
    monkeypatch.setattr(service_module, "load_artifacts", loader)

    path = str(tmp_path) if as_string else tmp_path

    service = service_module.RecommendationService.from_artifacts(path)

    assert service.artifacts is sentinel.artifacts
    loader.assert_called_once_with(str(tmp_path))
