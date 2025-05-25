import numpy as np
import pytest

from package.RankAMIP.data_script import simulate_bt_design_matrix
from package.RankAMIP.logistic import run_logistic_regression
from package.RankAMIP.logistic import isRankingRobust

@pytest.mark.parametrize("num_teams", [100])
@pytest.mark.parametrize("num_games", [5000])
@pytest.mark.parametrize("seed",      [42])
@pytest.mark.parametrize("k",         [2, 10, 50])
@pytest.mark.parametrize("alphaN",    [10, 100, 1000])
def test_isRankingRobust_various(num_teams, num_games, seed, k, alphaN):
    # 1) simulate data
    X, y = simulate_bt_design_matrix(num_teams, num_games, seed)

    # 2) run robustness check routine
    playerA, playerB, orig_out, new_out, indices = isRankingRobust(k, alphaN, X, y)

    # 3) fit on full data
    model_full = run_logistic_regression(X, y)
    orig_true  = model_full.coef_[0][playerA] - model_full.coef_[0][playerB]

    # 4) drop flagged games, refit
    Xd = np.delete(X, indices, axis=0)
    yd = np.delete(y, indices, axis=0)
    model_d  = run_logistic_regression(Xd, yd)
    new_true = model_d.coef_[0][playerA] - model_d.coef_[0][playerB]

    # 5) assert match
    assert pytest.approx(orig_out, abs=1e-9) == orig_true
    assert pytest.approx(new_out,  abs=1e-9) == new_true






