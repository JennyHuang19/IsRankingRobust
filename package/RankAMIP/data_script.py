import numpy as np
import pandas as pd

def make_BT_design_matrix(
    df: pd.DataFrame
) -> "tuple[np.array, np.array, dict]":
    '''
    Given a preference dataset, make it a logistic regression
    Arg:
        df: a pd.dataframe with first column being first team, second column to be second team and third indicating whether first team wins
    Return:
        X: design matrix X
        y: responses
        player_to_id: encoder of teams, with the 0th team to have a score of 0
    '''
    all_players = pd.concat([df.iloc[:, 0], df.iloc[:, 1]])
    all_players = pd.concat([df.iloc[:, 0], df.iloc[:, 1]])


    unique_players = all_players.unique()

    player_to_id = {player: idx for idx, player in enumerate(unique_players)}

    n_players = len(player_to_id)
    n_matches = df.shape[0]

    encoded_player1 = df.iloc[:, 0].map(player_to_id)
    encoded_player2 = df.iloc[:, 1].map(player_to_id)
    matches = np.arange(n_matches)
    X_tmp = np.zeros((n_matches, n_players))
    X_tmp[matches, encoded_player1] = 1
    X_tmp[matches, encoded_player2] = -1
    X = X_tmp[:,1:]
    y = np.array(df.iloc[:,2])
    return X, y, player_to_id

def simulate_bt_design_matrix(num_teams: int,
                              num_games: int,
                              seed: int = 42
                             ) -> 'tuple[np.ndarray, np.ndarray]':
    """
    Simulate pairwise match outcomes under a Bradley–Terry logistic model,
    fix beta_0 = 0, and build the corresponding design matrix X and labels y.

    Parameters
    ----------
    num_teams : int
        Total number of teams (>= 2).
    num_games : int
        Number of random pairwise matches to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (num_games, num_teams - 1)
        Design matrix: each row has +1 for the (winner>0) team index-1,
        –1 for the loser>0 team index-1, zero elsewhere.
    y : np.ndarray, shape (num_games,)
        Labels: 1 if the first-chosen team wins, 0 otherwise.
    """
    if num_teams < 2:
        raise ValueError("Need at least two teams.")
    
    rng = np.random.default_rng(seed)
    
    # 1) Draw latent strengths (beta_0 = 0, others random small)
    true_betas = np.concatenate([[0.], rng.normal(scale=0.1, size=num_teams - 1)])
    
    # 2) Simulate matches
    matchups = []
    for _ in range(num_games):
        i, j = rng.choice(num_teams, size=2, replace=False)
        diff = true_betas[i] - true_betas[j]
        p_i = 1 / (1 + np.exp(-diff))
        winner = i if rng.random() < p_i else j
        matchups.append((i, j, winner))
    
    # 3) Build X and y
    X = np.zeros((num_games, num_teams - 1), dtype=int)
    y = np.zeros(num_games, dtype=int)
    
    for idx, (i, j, winner) in enumerate(matchups):
        # label: did team i (the first-chosen) win?
        y[idx] = int(winner == i)
        
        # helper: map team k>0 to column k-1
        if winner == i:
            # +1 for i, -1 for j
            if i > 0:
                X[idx, i - 1] += 1
            if j > 0:
                X[idx, j - 1] -= 1
        else:
            # +1 for j, -1 for i
            if j > 0:
                X[idx, j - 1] += 1
            if i > 0:
                X[idx, i - 1] -= 1
    
    return X, y


def break_ties_randomly(df, seed=6):
    np.random.seed(seed)
    tie_indices = df[df['winner_tie'] == 1].index
    # Random choices: 1 means assign to model_a, 0 to model_b
    assign_to_a = np.random.choice([0, 1], size=len(tie_indices))
    
    for idx, assign in zip(tie_indices, assign_to_a):
        if assign == 1:
            df.at[idx, 'winner_model_a'] = 1
            df.at[idx, 'winner_model_b'] = 0
        else:
            df.at[idx, 'winner_model_a'] = 0
            df.at[idx, 'winner_model_b'] = 1
    
    return df
