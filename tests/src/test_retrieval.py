import pandas as pd
import pytest

from src.retrieval import RetrievalPipeline


def test_recommend_before_fit():
    pipeline = RetrievalPipeline({}, {}, {})

    with pytest.raises(RuntimeError, match="fitted first"):
        pipeline.make_candidates([10])


def test_real_pipeline_returns_original_ids_and_filters_seen(
        fitted_retrieval, raw_dataset
):
    history = raw_dataset[2]
    result = fitted_retrieval.make_candidates([20, 37, 20])

    assert set(result.user_id) == {20, 37}
    assert not result.duplicated(["user_id", "movie_id"]).any()
    assert result.groupby("user_id").size().max() <= fitted_retrieval.candidate_k

    for user_id, rows in result.groupby("user_id"):
        watched = set(history.loc[history.user_id == user_id, "movie_id"])

        assert not (set(rows.movie_id) & watched)
        assert rows.rrf.is_monotonic_decreasing

    assert set(result.movie_id) <= set(history.movie_id)


def test_unknown_user_is_explicitly_rejected(fitted_retrieval):
    with pytest.raises(ValueError, match="Unknown user IDs"):
        fitted_retrieval.make_candidates([9999999])


def test_refit_rebuilds_mappings_and_does_not_modify_input(
        fitted_retrieval, raw_dataset
):
    history = raw_dataset[2]
    subset = history.loc[history.user_id.isin([54, 71])].copy()
    original = subset.copy()

    subset["user_idx"] = 1000
    fitted_retrieval.fit(subset)

    assert set(fitted_retrieval.data.user_to_idx) == {54, 71}
    assert fitted_retrieval.data.user_item.shape[0] == 2
    assert subset.user_idx.eq(1000).all()
    pd.testing.assert_frame_equal(subset.drop(columns="user_idx"), original)

    result = fitted_retrieval.make_candidates([54])

    assert set(result.user_id) == {54}
    with pytest.raises(ValueError):
        fitted_retrieval.make_candidates([20])