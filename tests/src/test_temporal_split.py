import pandas as pd
import pytest

from src.data.temporal_split import TemporalSplit


@pytest.mark.parametrize("method", ["split_to_two", "split_to_three"])
def test_split_is_chronological_and_preserves_every_row(method):
    raw = pd.DataFrame(
        [
            {"event_id": f"{u}-{t}", "user_id": u, "timestamp": t}
            for u in [20, 100]
            for t in range(8)
        ]
    ).sample(frac=1, random_state=42)

    original = raw.copy()
    parts = getattr(TemporalSplit(), method)(raw, train_size=0.5)

    all_ids = [time for part in parts for time in part["event_id"]]

    assert len(all_ids) == len(set(all_ids)) == len(raw)
    assert set(all_ids) == set(raw["event_id"])

    expected = [list(range(4)), list(range(4, 8))]

    if method == "split_to_three":
        expected = [list(range(4)), [4, 5], [6, 7]]

    for part, timestamps in zip(parts, expected, strict=True):
        assert part.index.tolist() == list(range(len(part)))

        for user_id in [20, 100]:
            assert part.loc[part.user_id == user_id, "timestamp"].tolist() == timestamps

    pd.testing.assert_frame_equal(raw, original)


@pytest.mark.parametrize("method", ["split_to_two", "split_to_three"])
@pytest.mark.parametrize("train_size", [0, 1, -0.5, 1.5])
def test_invalid_split_fraction_is_rejected(method, train_size, history):
    with pytest.raises(ValueError):
        getattr(TemporalSplit(), method)(history, train_size=train_size)


@pytest.mark.parametrize("method", ["split_to_two", "split_to_three"])
def test_empty_data_is_rejected(method, history):
    with pytest.raises(ValueError):
        getattr(TemporalSplit(), method)(history.iloc[:0])