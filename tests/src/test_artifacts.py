import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from src.artifacts import load_artifacts, save_artifacts


def test_save_load_ranker_retrieval_and_metadata(
    fitted_retrieval, raw_dataset, fast_config, tmp_path
):
    users, movies, history = raw_dataset

    X = pd.DataFrame(
        {
            "z_test": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "a_test": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )

    pool = Pool(X, label=[0, 1, 2, 0, 2, 1], group_id=[0, 0, 0, 1, 1, 1])

    ranker = CatBoostRanker(
        iterations=2,
        depth=2,
        loss_function="YetiRank",
        task_type="CPU",
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    ranker.fit(pool)

    fast_config.CAT_FEATURES = []

    before = fitted_retrieval.make_candidates([20, 37])
    directory = tmp_path / "nested" / "model"

    save_artifacts(
        directory=str(directory),
        ranker=ranker,
        retrieval=fitted_retrieval,
        users=users,
        movies=movies,
        history=history,
        feature_columns=list(X.columns),
        config=fast_config,
        metrics={"recall@10": 0.25}
    )

    loaded = load_artifacts(str(directory))
    after = loaded["retrieval"].make_candidates([20, 37])

    pd.testing.assert_frame_equal(
        before, after
    )

    np.testing.assert_allclose(
        ranker.predict(X), loaded["ranker"].predict(X)
    )

    assert loaded["metadata"]["feature_columns"] == ["z_test", "a_test"]
    assert loaded["metadata"]["evaluation_metrics"] == {"recall@10": 0.25}

    for key, original in [("users", users), ("movies", movies), ("history", history)]:
        pd.testing.assert_frame_equal(loaded[key], original)

    data = loaded["retrieval"].data

    np.testing.assert_array_equal(
        data.user_item.toarray(), fitted_retrieval.data.user_item.toarray()
    )

    assert data.user_to_idx == fitted_retrieval.data.user_to_idx
    assert data.movie_to_idx == fitted_retrieval.data.movie_to_idx