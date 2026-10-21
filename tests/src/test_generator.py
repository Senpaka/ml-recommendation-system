from unittest.mock import Mock

import numpy as np
import pytest

from src.data.preprocessing import build_retrieval_data
from src.generator import CandidateGenerator


def make_generator(history, limit=10):
    sources = [Mock(), Mock(), Mock()]

    sources[0].recommend.return_value = [(500, 0.9), (300, 0.8)]
    sources[1].recommend.return_value = [(300, 0.7), (900, 0.6)]
    sources[2].recommend.return_value = [(300, 4.5)]

    data = build_retrieval_data(history=history)

    generator = CandidateGenerator(
        *sources,
        data=data,
        k_als=2,
        k_sim=3,
        k_pop=4,
        candidate_k=limit,
    )

    return generator, sources, data


def test_sources_merge_into_one_row_with_rrf(history):

    generator, sources, data = make_generator(history=history)

    idx = data.user_to_idx[100]
    result = generator.build_candidates(np.array([idx]))

    assert len(result) == 3
    assert result["movie_id"].is_unique
    assert result["user_id"].tolist() == [100, 100, 100]
    assert result.iloc[0]["movie_id"] == 300

    row = result.set_index("movie_id").loc[300]

    assert row["rrf"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert row[["als_rank", "sim_rank", "pop_rank"]].tolist() == [2, 1, 1]
    assert row[["from_als", "from_sim", "from_pop"]].tolist() == [1, 1, 1]

    absent = result.set_index("movie_id").loc[900]

    assert absent[["als_score", "als_rank", "from_als"]].tolist() == [0, 0, 0]

    for source, k in zip(sources, [2, 3, 4], strict=True):
        source.recommend.assert_called_once_with(idx, k=k)


def test_limit_is_applied_per_user(history):

    generator, _, data = make_generator(history=history, limit=1)

    result = generator.build_candidates(
        np.array([data.user_to_idx[20], data.user_to_idx[100]])
    )

    assert result.groupby("user_id").size().to_dict() == {20: 1, 100: 1}
    assert result["movie_id"].tolist() == [300, 300]


def test_empty_sources_return_empty_frame(history):
    generator, sources, _ = make_generator(history)

    for source in sources:
        source.recommend.return_value = []

    assert generator.build_candidates(np.array([0])).empty
    assert generator.build_candidates(np.array([], dtype=int)).empty
