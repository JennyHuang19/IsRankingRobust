import numpy as np
import pytest

# from package.RankAMIP.logistic import find_closest_matchups
def find_closest_matchups(scores: np.ndarray, K: int) -> 'list[tuple[int,int,float]]':
    """
    For each top-index t in [0..K-1] and each rest-index r in [K..P-1],
    compute (t, r, scores[t] - scores[r]) and return as a list.
    """
    P = scores.shape[0]
    top  = scores[:K]         # shape (K,)
    rest = scores[K:]         # shape (P-K,)

    # diffs[t, r-K] = scores[t] - scores[r]
    diffs = top[:, None] - rest[None, :]  # shape (K, P-K)

    # build flat index arrays of length K*(P-K)
    t_idx = np.repeat(np.arange(K), P - K)  # [0,0,…,1,1,…,K-1, …]
    r_idx = np.tile(np.arange(K, P), K)  # [K,K+1,…,K,K+1,…, …]

    matchups = list(zip(
        t_idx.tolist(),
        r_idx.tolist(),
        diffs.ravel().tolist()
    ))
    # sort the matchups by the difference.
    sorted_matchups = sorted(matchups, key=lambda x: x[2])
    
    return sorted_matchups

def test_single_pair():
    scores = np.array([10.0, 5.0])
    # Only one top (0) vs rest (1): diff = 10−5 = 5
    expected = [(0, 1, 5.0)]
    assert find_closest_matchups(scores, K=1) == expected

def test_simple_ordering():
    scores = np.array([5.0, 3.0, 1.0])
    # Top (0) vs rest (1,2): diffs = [2.0, 4.0], sorted already
    expected = [(0, 1, 2.0), (0, 2, 4.0)]
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