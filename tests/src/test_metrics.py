import numpy as np
import pytest

from src.metrics.metrics import ndcg_at_k, precision_at_k, recall_at_k


@pytest.mark.parametrize(
    "metric, expected",
    [
        (precision_at_k, 0.5),
        (recall_at_k, 1 / 3),
    ],
)
def test_hits_are_counted_with_correct_denominator(metric, expected):
    assert metric(np.array([10, 20, 30]), np.array([10, 99, 20]), k=2) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "truth, predictions, k, expected",
    [
        ([10, 20], [20, 10, 99], 3, 1.0),
        ([10], [99, 10], 2, 1 / np.log2(3)),
        ([10, 20], [10], 10, 1 / (1 + 1 / np.log2(3))),
        ([10], [99, 10], 1, 0.0),
    ],
)
def test_ndcg_values(truth, predictions, k, expected):
    assert ndcg_at_k(np.array(truth), np.array(predictions), k) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
@pytest.mark.parametrize("truth, predictions", [([10], []), ([], [10]), ([10], [99])])
def test_no_hits_return_zero(metric, truth, predictions):
    assert metric(np.array(truth), np.array(predictions), k=10) == 0.0


@pytest.mark.parametrize("metric", [precision_at_k, recall_at_k, ndcg_at_k])
@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(metric, k):
    with pytest.raises(ValueError):
        metric(np.array([10]), np.array([10]), k=k)
