import numpy as np
import pandas as pd
import pytest
from catboost import Pool

from src.artifacts import load_artifacts
from src.data.preprocessing import preprocess_data
from src.features.features import build_features
from src.train import _evaluate_ranker


class FixedRanker:
    def predict(self, pool):
        assert pool.get_feature_names() == ["signal"]
        return np.array([0.1, 0.9])


def test_evaluation_includes_users_without_predictions():
    data = pd.DataFrame({"user_id": [20, 20], "movie_id": [500, 900], "signal": [0, 1]})

    target = pd.DataFrame(
        {
            "user_id": [20, 100],
            "movie_id": [900, 300],
            "rating": [5, 4]
        }
    )

    metrics = _evaluate_ranker(FixedRanker(), data, target, ["signal"], [], k=10)

    assert metrics == {"recall@10": 0.5, "ndcg@10": 0.5, "evaluated_users": 2}


def test_evaluation_rejects_target_without_positives():
    data = pd.DataFrame({"user_id": [20, 20], "movie_id": [500, 900], "signal": [0, 1]})
    target = pd.DataFrame({"user_id": [20], "movie_id": [900], "rating": [2]})

    with pytest.raises(ValueError, match="No positive interactions"):
        _evaluate_ranker(FixedRanker(), data, target, ["signal"], [], k=10)


def test_train_cpu_saves_loadable_working_artifacts(
    raw_dataset, fast_config, tmp_path, monkeypatch
):
    import src.train as training

    monkeypatch.setattr(training, "Config", lambda: fast_config)
    monkeypatch.chdir(tmp_path)

    model, metrics = training.train(*raw_dataset)

    assert model.tree_count_ == 3
    assert metrics["evaluated_users"] == 8
    assert 0 <= metrics["recall@10"] <= 1
    assert 0 <= metrics["ndcg@10"] <= 1

    artifacts = load_artifacts(str(tmp_path / "artifacts/current"))
    expected_users, expected_movies, expected_history = preprocess_data(*raw_dataset)

    retrieval = artifacts["retrieval"]
    seen = retrieval.data.seen_movies[retrieval.data.user_to_idx[20]]
    candidates = retrieval.make_candidates([20])
    data = build_features(
        candidates, artifacts["history"], artifacts["users"], artifacts["movies"]
    )
    columns = artifacts["metadata"]["feature_columns"]

    pd.testing.assert_frame_equal(artifacts["users"], expected_users)
    pd.testing.assert_frame_equal(artifacts["movies"], expected_movies)
    pd.testing.assert_frame_equal(artifacts["history"], expected_history)
    assert artifacts["metadata"]["evaluation_metrics"] == metrics
    assert artifacts["metadata"]["n_users"] == 8
    assert artifacts["metadata"]["n_movies"] == 24
    assert set(retrieval.data.user_to_idx) == set(expected_users.user_id)
    assert seen == set(expected_history.loc[expected_history.user_id == 20, "movie_id"])
    assert not candidates.empty
    assert set(columns).isdisjoint({"label", "user_id", "user_idx", "movie_id", "group_id"})

    pool = Pool(data[columns], cat_features=artifacts["metadata"]["cat_features"])

    before = model.predict(pool)
    after = artifacts["ranker"].predict(pool)

    assert np.isfinite(after).all()
    np.testing.assert_allclose(after, before)