import numpy as np
import pytest
from RankAMIP.logistic import find_closest_matchups



def test_single_pair():
    scores = np.array([10.0, 5.0])
    # Only one top (0) vs rest (1): diff = 10−5 = 5
    expected = [(0, 1, 5.0)]
    breakpoint()
    assert find_closest_matchups(scores, K=1) == expected

def test_simple_ordering():
    scores = np.array([3.0, 5.0,  1.0])
    # Top (0) vs rest (1,2): diffs = [2.0, 4.0], sorted already
    expected = [(0, 1, 2.0), (0, 2, 4.0)]
    breakpoint()
    assert find_closest_matchups(scores, K=1) == expected

def test_multiple_top_indices():
    scores = np.array([7.0, 5.0, 3.0, 1.0])
    # Top indices 0,1 vs rest 2,3:
    # (0,2,4),(0,3,6),(1,2,2),(1,3,4) → sorted by diff
    expected = [
        (1, 2, 2.0),
        (0, 2, 4.0),
        (1, 3, 4.0),
        (0, 3, 6.0),
    ]
    assert find_closest_matchups(scores, K=2) == expected


def test_empty_when_K_zero_or_full():
    scores = np.array([1.0, 2.0, 3.0])
    # K=0 → no top, K=3 → no rest
    assert find_closest_matchups(scores, K=0) == []
    assert find_closest_matchups(scores, K=3) == []


def test_diff_type_and_correctness():
    scores = np.array([5.5, 2.2, 1.1, 0.0])
    result = find_closest_matchups(scores, K=2)
    # Expect 2*(4−2)=4 entries
    assert len(result) == 4
    for t, r, diff in result:
        assert isinstance(t, int)
        assert isinstance(r, int)
        assert isinstance(diff, float)
        # Check numeric correctness
        assert pytest.approx(diff) == scores[t] - scores[r]
test_simple_ordering()