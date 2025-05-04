import numpy as np 
from RankAMIP.logistic import make_influence_wrt_player
from RankAMIP.logistic import run_logistic_regression
from RankAMIP.logistic import make_AMIP_sign_change_playerij
 
# Set seed for reproducibility
np.random.seed(42)

# --- Step 1: Define teams and latent strengths ---
teams = ['A', 'B', 'C', 'D', 'E', 'F']
team_ids = {name: i for i, name in enumerate(teams)}
true_betas = np.array([0.05, 0.04, 0.03, -0.00, -0.01, -0.02])  # (note: we should try out different signal levels later).


n_games = 10000
matchups = []

for _ in range(n_games):
    i, j = np.random.choice(6, size=2, replace=False) # randomly choose 2 teams to compete.
    beta_diff = true_betas[i] - true_betas[j]
    prob_win_i = 1 / (1 + np.exp(-beta_diff))
    winner = i if np.random.rand() < prob_win_i else j
    matchups.append((i, j, winner))


X = np.zeros((n_games, 5))  # We fix beta_0 = 0 and estimate beta_1 and beta_2.
y = np.zeros(n_games)

for idx, (i, j, winner) in enumerate(matchups):
    # Map to reduced index space (beta_0 = 0)
    def reduced(k): return k - 1 if k > 0 else None
    
    if winner == i:
        y[idx] = 1
        if reduced(i) is not None:
            X[idx, reduced(i)] += 1
            # print(f"i, reduced(i): {i, reduced(i)}")
        if reduced(j) is not None:
            X[idx, reduced(j)] -= 1
    else:
        y[idx] = 0
        if reduced(j) is not None:
            X[idx, reduced(j)] += 1
        if reduced(i) is not None:
            X[idx, reduced(i)] -= 1
'''

X = np.random.rand(10000, 6) * 2 -1
X_beta = X @ true_betas.T
probs = 1/(1+np.exp(-X_beta))

y = (np.random.rand(n_games) <= probs) * 1.
#breakpoint()
#cache = {}
'''
full_model = run_logistic_regression(X, y)
full_model.coef_[0]
pos_p_hats = full_model.predict_proba(X)[:, 1]

influence_fun = make_influence_wrt_player(pos_p_hats, X, y, "IF")


influence_first = influence_fun(0)
betas = full_model.coef_[0]
#breakpoint()

#######

amip_ij = make_AMIP_sign_change_playerij(betas, 10, pos_p_hats, X, y, "1sN", refit = True)
change_sign_amip, change_sign_refit, new_beta_diff_amip, new_beta_diff_refit, idx = amip_ij(2, 0)

#breakpoint()