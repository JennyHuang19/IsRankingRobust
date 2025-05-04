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
                        method: str = "1sN") -> np.ndarray:
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
            cache[dim] = res
            return res
        
    return get_influence




def compute_approx(
    pos_p_hats: np.ndarray,
    X: np.ndarray,
    index: int,
    y: np.ndarray,
    method: str,
    e: np.ndarray,
) -> float:
    """
    pos_p_hats: np.array, shape (n,), the predicted probabilities.
    X: np.array, shape (n, p), the design matrix.
    index: int, the index of the data point whose influence we want to compute.
    y: np.array, shape (n,), the response variable.
    method: str, the method to use to compute the approximation ("1sN" or "IF").
    e: np.ndarray, shape (p,), the direction of the influence function.

    Compute the influence function approximation of
    the effect of infinitesimally upweighting the index-th
    data point on a quantity of interest 
    (e.g., some linear combination of the 
    logistic regression coefficients,
    determined by the choice of e).
    """
    v_lst = pos_p_hats * (1 - pos_p_hats)
    V = np.diag(v_lst)
    if method == "IF":
        influence_function = (
            e @ np.linalg.inv(X.T @ V @ X) @ X[index] * (y[index] - pos_p_hats[index])
        ) # solve linear system rather than inverting matrix.
        return influence_function[0]
    elif method == "1sN":
        influence_function = (
            e @ np.linalg.inv(X.T @ V @ X) @ X[index] * (y[index] - pos_p_hats[index])
        )
        h_ii = compute_leverage(pos_p_hats, X, index, y)
        return 1 / (1 - h_ii) * influence_function[0]
    else:
        return "Invalid method."


def compute_approx_multiD(
    pos_p_hats: np.ndarray,
    X: np.ndarray,
    index: int,
    y: np.ndarray,
    e: np.ndarray,
    method: str,
) -> float:
    """
    pos_p_hats: np.array, shape (n,), the predicted probabilities.
    X: np.array, shape (n, p), the design matrix.
    index: int, the index of the data point whose influence we want to compute.
    y: np.array, shape (n,), the response variable.
    e: np.array, shape (n,), the direction of interest (e.g. [1, 0, 0, 0, 0]).
    method: str, the method to use to compute the approximation ("1sN" or "IF").

    Compute the influence function approximation of
    the effect of infinitesimally upweighting the index-th
    data point on the logistic regression coefficient.
    """
    v_lst = pos_p_hats * (1 - pos_p_hats)
    V = np.diag(v_lst)
    if method == "IF":
        influence_function = (
            e @ np.linalg.inv(X.T @ V @ X) @ X[index] * (y[index] - pos_p_hats[index])
        )
        return influence_function
    elif method == "1sN":
        influence_function = (
            e @ np.linalg.inv(X.T @ V @ X) @ X[index] * (y[index] - pos_p_hats[index])
        )
        h_ii = compute_leverage(pos_p_hats, X, index, y)
        return 1 / (1 - h_ii) * influence_function
    else:
        return "Invalid method."


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


def run_experiment_generate_dataframe(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    X_position: np.ndarray,
) -> pd.DataFrame:
    """
    X_orig: np.array, shape (n, p), the design matrix without the outlier.
    y_orig: np.array, shape (n,), the response variable without the outlier.
    X_position: np.array, shape (n, p), the array of x-positions of outliers.

    returns: a dataframe that stores the summary statistics for the regression
    run on the particular outlier.
    """
    summary_list = []
    # run model without the outlier.
    model_inliers = run_logistic_regression(X_orig, y_orig)

    for i in range(len(X_position)):
        # add d_n to the end of the data.
        X_new = np.vstack((X_orig, X_position[i]))
        y_new = np.append(y_orig, 0)

        # fit a logistic regression model
        model = run_logistic_regression(X_new, y_new)
        # compute the predicted probabilities
        pos_p_hats = model.predict_proba(X_new)[:, 1]
        # plot the data
        # plot_data(X_new, y_new, model, 20)

        # print components of the leverage
        leverage = compute_leverage(pos_p_hats, X_new, len(X_new) - 1, y_new)
        if_func = compute_approx(pos_p_hats, X_new, len(X_new) - 1, y_new, "IF")
        one_sN = compute_approx(pos_p_hats, X_new, len(X_new) - 1, y_new, "1sN")
        delta = model.coef_[0][0] - model_inliers.coef_[0][0]

        summary_dict = {
            "X": X_position[i][0],
            "IF": if_func,  # if approx to change in fit.
            "1sN": one_sN,  # 1sN approx to change in fit.
            "delta": delta,  # true change in fit
            "new_fit": model.coef_[0][0],  # new fit
            "orig_fit": model_inliers.coef_[0][0],  # original fit
            "leverage": leverage,  # leverage of the outlier data point.
            "residual": y_new[-1]
            - pos_p_hats[-1],  # residual of the outlier data point.
            "p_hat": pos_p_hats[-1],  # p_hat of the outlier data point.
            "p_hat_one_minus_p_hat": pos_p_hats[-1] * (1 - pos_p_hats[-1]),
            "X_2_p_hat_one_minus_p_hat_lst": (
                X_new[-1] ** 2 * pos_p_hats[-1] * (1 - pos_p_hats[-1])
            )[0],
            "inv_xtvx": (
                np.linalg.inv(X_new.T @ np.diag(pos_p_hats * (1 - pos_p_hats)) @ X_new)
            )[0][0],
        }
        summary_list.append(summary_dict)

    return pd.DataFrame(summary_list)