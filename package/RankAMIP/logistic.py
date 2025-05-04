import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression




def make_BT_design_matrix(
    df: pd.DataFrame
) -> tuple[np.array, np.array, dict]:
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


def run_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = False,
    penalty: str = None,
) -> LogisticRegression:
    """
    Fit a logistic regression model.
    """
    model = LogisticRegression(fit_intercept=fit_intercept, penalty=penalty)
    model.fit(X, y)
    return model



def make_influence_wrt_player(pos_p_hats: np.ndarray,
                        X: np.ndarray,
                        y: np.ndarray,
                        method: str = "1sN") -> callable:
    """
    Compute the influence of each data point on the j-th regression coefficient.

    Parameters
    ----------
    pos_p_hats : np.ndarray, shape (n,)
        Predicted probabilities p_i from the fitted logistic model.
    X : np.ndarray, shape (n, p)
        Design matrix.
    y : np.ndarray, shape (n,)
        Binary responses (0/1).
    method : {"IF", "1sN"}
        - "IF":     influence function
        - "1sN":    one-step Newton (scale by 1/(1−h_i))

    Returns
    -------
    get_influence: callable function query in influence 
            returns np.ndarray, shape (n,)
            Influence scores of each point on coefficient `dim`.
    """
    n, p = X.shape
    if method != "IF" and method != "1sN":
        raise ValueError("method must be 'IF' or '1sN'")
    

    # 1) Hessian and its inverse
    v = pos_p_hats * (1 - pos_p_hats)            # (n,)
    H = X.T @ (v[:, None] * X)                  # (p, p)
    # print("v[:, None]", v[:, None])
    invH = np.linalg.inv(H)                     # (p, p)

    # 2) residuals
    resid = (y - pos_p_hats)                    # (n,)

    # 3) unscaled influence on dim: r_i * (X_i @ invH[:, dim])
    #    build p-vector direction once:
    cache = {}

    def get_influence(dim):
        '''
        query influence score at dim and updating cache
        Arg:
            dim: int between 0 and p
            cache: a dict with key being the parameter index and value being a np.array of influence score
        return:
            influence scores
        '''

        if not (0 <= dim < p):
            raise IndexError(f"dim must be in [0, {p}), got {dim}")
        if dim in cache:
            return cache[dim]
        
        invH_col = invH[:, dim]                     # (p,)
        #    then each row X_i dot d gives a length-n vector
        influence_unscaled = resid * (X @ invH_col) # (n,)
        
        if method == "IF":
            cache[dim] = influence_unscaled
            return influence_unscaled

        elif method == "1sN":
            # compute leverages h_i
            Hprod = X @ invH                         # (n, p)
            h = v * np.einsum("ij,ij->i", Hprod, X)  # (n,)
            res = influence_unscaled / (1.0 - h)
            #breakpoint()
            cache[dim] = res
            return res
        
    return get_influence


def make_AMIP_sign_change_playerij(beta: np.array,
                       alphaN: int,
                       pos_p_hats: np.ndarray,
                       X: np.ndarray,
                       y: np.ndarray,
                       method: str = "1sN", 
                       refit: bool = False, 
                       refit_config = {}
                       ):
    '''
    make function that test if beta_i-beta_j is robust to data dripping
    Args:
        beta: the fitted beta
        alphaN: the number to check robust to dropping 
        pos_p_hats: predicted winning rate 
        X: design 
        y: original responses
        method: what to use to approximate data dropping, has to be {1sN, IF}
        refit: if we refit the model with identified MIS data dropped
        refit_config: a dict with refitting parameters
    return: 
        amip_playerij: callable, taking index i and j return if beta_i-beta_j is robust, if j is None, return if beta_i can change sign
            Arg: i and j, int
            Return:
                robust_or_not: whether the sign of beta_i-beta_j (or beta_i if j is None) is robust to dropping alphaN data, bool
                amip_refit: amip approximation or refit, float
                new_beta_diff_refit: actual refit, float

    
    '''
    get_influence = make_influence_wrt_player(pos_p_hats, X, y, method)
    def amip_playerij(dim_1, dim_2 = None):
        if dim_2 is None: # this is useful when comparing to reference level 0
            beta_i = beta[dim_1]
            influence = -get_influence(dim_1)
            top = np.argsort(influence)
            if beta_i < 0:
                top = top[::-1]
            change = np.sum(influence[top[:alphaN]])
            new_betai_amip = beta_i + change
            change_sign_amip = np.sign(new_betai_amip) != np.sign(beta_i)
            if refit:
                res = run_logistic_regression(X[top[alphaN:,]], 
                                              y[top[alphaN:]],
                                              *refit_config)
                new_betai_refit = res.coef_[0][dim_1]
                change_sign_refit = np.sign(new_betai_refit) != np.sign(beta_i)
            else:
                new_betai_refit = None
                change_sign_refit = None

            #breakpoint()
            return change_sign_amip, change_sign_refit, new_betai_amip, new_betai_refit, top[:alphaN]

        beta_diff = beta[dim_1] - beta[dim_2]

        influence = -(get_influence(dim_1) - get_influence(dim_2))
        top = np.argsort(influence)
        if beta_diff < 0: # if beta is negative, we want the positive part of the influence score
            top = top[::-1]
        #breakpoint()
        change = np.sum(influence[top[:alphaN]])
        new_beta_diff_amip = beta_diff + change
        change_sign_amip = np.sign(new_beta_diff_amip) != np.sign(beta_diff)
        

        if refit:
            res = run_logistic_regression(X[top[alphaN:,]], 
                                              y[top[alphaN:]],
                                              *refit_config)
            new_beta_diff_refit = res.coef_[0][dim_1] - res.coef_[0][dim_2]
            change_sign_refit = np.sign(new_beta_diff_refit) != np.sign(beta_diff)
        else:
            new_beta_diff_refit = None
            change_sign_refit = None
        return change_sign_amip, change_sign_refit, new_beta_diff_amip, new_beta_diff_refit, top[:alphaN]

    return amip_playerij

                







def compute_leverage(
    pos_p_hats: np.ndarray,
    X: np.ndarray,
    index: int,
    y: np.ndarray,
) -> float:
    """
    pos_p_hats: np.array, shape (n,), the predicted probabilities.
    X: np.array, shape (n, p), the design matrix.
    index: int, the index of the data point whose influence we want to compute.
    y: np.array, shape (n,), the response variable.

    Compute the leverage of the index-th data point.
    """
    v_lst = pos_p_hats * (1 - pos_p_hats)
    V = np.diag(v_lst)
    H = V @ X @ np.linalg.inv(X.T @ V @ X) @ X.T
    return H[index, index]

