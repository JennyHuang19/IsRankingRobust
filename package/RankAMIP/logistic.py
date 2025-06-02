import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression



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

def find_closest_matchups(player_scores: np.ndarray, k: int) -> 'list[tuple[int,int,float]]':
    """
    For each top-index t in [0..k-1] and each rest-index r in [k..P-1],
    compute (t, r, player_scores[t] - player_scores[r]) and return as a list.
    """
    P = player_scores.shape[0] + 1
    #breakpoint()
    full_score = np.concatenate((np.array([0]), player_scores))
    asort = np.argsort(full_score)[::-1] # players sorted from big to small

    matchups = []
    for i in range(k):
        for j in range(P-k):
            diff = np.abs(full_score[asort[i]]-full_score[asort[j+k]]).item()
            tm1 = asort[i].item()-1
            tm2 = asort[j+k].item()-1
            if tm1 == -1:
                matchups.append((tm2, None, diff))
            elif tm2 == -1:
                matchups.append((tm1, None, diff))
            else:
                matchups.append((tm1, tm2, diff))

    sorted_matchups = sorted(matchups, key=lambda x: x[2])
    #breakpoint()
    return sorted_matchups


def isRankingRobust(k, alphaN, X, y):
    '''
    Checks if the ranking of the top k players/models is robust to data-dropping.
    Arg: 
        k, int, number of top players to consider. 
        alphaN, int, amount of data willing to drop.
        X, np.ndarray, design matrix.
        y, np.ndarray, response vector.
    Return:
        playerA, playerB: int, indices of players/models.
        new_beta_diff_refit: float, new beta difference.
        indices: list, indices of dropped data.
    '''
    # run logistic regression on X, y
    myAMIP = LogisticAMIP(X, y, fit_intercept=False, penalty=None)
    player_scores = myAMIP.model.coef_[0] # (p,)

    
    close_matchups = find_closest_matchups(player_scores, k)
    for playerA, playerB, diff in close_matchups: # a list of k(p-k) matchups.
        # print("testing new matchup: ", playerA, playerB)
        sign_change_amip, sign_change_refit, original_beta_diff, new_beta_diff_amip, new_beta_diff_refit, indices = myAMIP.AMIP_sign_change(alphaN, playerA, playerB)
        if sign_change_refit:
            return playerA, playerB, original_beta_diff, new_beta_diff_refit, indices
    
    return -1, -1, -1, -1, [-1] # when ranking is robust.

class LogisticAMIP():
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 fit_intercept: bool = False, 
                 penalty: str = None
                 ):
        '''
        Class for dealing with AMIP in logistic regression
        Args:
            X: design matrix 
            y: responses, binary
            fit_intercept: bool, whether to fit intercept
            penalty: bool whether to have penalty
            refit: bool, whether to refit when approximating dropping data
        '''
        self.X = X
        self.y = y
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.model = run_logistic_regression(X, y, fit_intercept, penalty)
        self.pos_p_hats = self.model.predict_proba(X)[:, 1]

        ### private stuff ###
        self.__IFcache__ = {}
        self.__oneSNcache__ = {}
        self.__n__ = X.shape[0]
        self.__p__ = X.shape[1]
        

        self.__v__ = self.pos_p_hats * (1 - self.pos_p_hats) # (n,)
        H = X.T @ (self.__v__[:, None] * X)                  # (p, p)
        self.__invH__ = np.linalg.inv(H)                     # (p, p)
        self.__resid__ = (y - self.pos_p_hats)  
        self.__Hprod__ = self.X @ self.__invH__
    
    def get_influence_IF(self, dim):
        '''
        get influence approximation with influence function
        Args:
            dim: int the parameter to calculate influence approximation on
        Return:
            res: nd.array, influence approximation for all data points
        '''
        if not (0 <= dim < self.__p__):
            raise IndexError(f"dim must be in [0, {self.__p__}), got {dim}")
        if dim in self.__IFcache__:
            return self.__IFcache__[dim]
        
        invH_col = self.__invH__[:, dim]                     # (p,)
        #    then each row X_i dot d gives a length-n vector
        influence_unscaled = self.__resid__ * (self.X @ invH_col) # (n,)
        self.__IFcache__[dim] = influence_unscaled
        return influence_unscaled
    
    def get_influence_1sN(self, dim):
        '''
        get influence approximation with 1 step Newton
        Args:
            dim: int the parameter to calculate influence approximation on
        Return:
            res: nd.array, influence approximation for all data points
        '''
        if not (0 <= dim < self.__p__):
            raise IndexError(f"dim must be in [0, {self.__p__}), got {dim}")
        if dim in self.__oneSNcache__:
            return self.__oneSNcache__[dim]
        invH_col = self.__invH__[:, dim]                     # (p,)
        #    then each row X_i dot d gives a length-n vector
        influence_unscaled = self.__resid__ * (self.X @ invH_col) # (n,)                         # (n, p)
        h = self.__v__ * np.einsum("ij,ij->i", self.__Hprod__, self.X)  # (n,)
        res = influence_unscaled / (1.0 - h)
        #breakpoint()
        self.__oneSNcache__[dim] = res
        return res



    def AMIP_sign_change(self, alphaN, dim_1, dim_2=None, 
                     method="1sN", refit=True, contains_ties=True,
                     SCALE=400, INIT_RATING=1000):
        '''
        This function uses AMIP to detect the sign change of a parameter or difference between two parameters, using ELO-scaled coefficients

        Args:
            alphaN: int, number of points to drop
            dim_1: int, first parameter index
            dim_2: int or None, second parameter index
            method: str, "1sN" or "IF"
            refit: bool, whether to refit
            contains_ties: bool, whether to handle row duplication
            SCALE: float, scaling multiplier for ELO
            INIT_RATING: float, ELO intercept shift

        Returns:
            change_sign_amip: bool
            change_sign_refit: bool
            beta_diff: float
            new_beta_amip: float
            new_beta_refit: float or None
            index: np.ndarray of dropped indices
        '''
        if method == "1sN":
            get_influence = self.get_influence_1sN
        elif method == "IF":
            get_influence = self.get_influence_IF
        else:
            raise ValueError("method has to be '1sN' or 'IF'")
    

        if contains_ties:
            nonWeightedX = self.X[::2]
            nonWeightedY = self.y[::2]
            res_full = run_logistic_regression(
                nonWeightedX, nonWeightedY,
                fit_intercept=self.fit_intercept,
                penalty=self.penalty
            )
            beta = res_full.coef_[0]
        else:
            beta = self.model.coef_[0]

        if dim_2 is None:
            beta_i = SCALE * beta[dim_1] + INIT_RATING
            influence = -SCALE * get_influence(dim_1)

            if contains_ties:
                if len(influence) % 2 != 0:
                    raise ValueError("Expected even length influence for duplicated rows")
                influence = influence.reshape(-1, 2).sum(axis=1)

            top = np.argsort(influence)
            if beta_i < INIT_RATING:
                top = top[::-1]

            change = np.sum(influence[top[:alphaN]])
            new_betai_amip = beta_i + change
            change_sign_amip = np.sign(new_betai_amip - INIT_RATING) != np.sign(beta_i - INIT_RATING)

            if refit:
                res = run_logistic_regression(
                    nonWeightedX[top[alphaN:],], nonWeightedY[top[alphaN:]],
                    fit_intercept=self.fit_intercept,
                    penalty=self.penalty
                )
                new_betai_refit = SCALE * res.coef_[0][dim_1] + INIT_RATING
                change_sign_refit = np.sign(new_betai_refit - INIT_RATING) != np.sign(beta_i - INIT_RATING)
            else:
                new_betai_refit = None
                change_sign_refit = None

            return change_sign_amip, change_sign_refit, beta_i, new_betai_amip, new_betai_refit, top[:alphaN]

        # Case where comparing two coefficients
        beta_diff = SCALE * (beta[dim_1] - beta[dim_2])
        influence_dim1 = get_influence(dim_1)
        influence_dim2 = get_influence(dim_2)

        if contains_ties:
            if len(influence_dim1) % 2 != 0 or len(influence_dim2) % 2 != 0:
                raise ValueError("Expected even length influences for duplicated rows")
            influence_dim1 = influence_dim1.reshape(-1, 2).sum(axis=1)
            influence_dim2 = influence_dim2.reshape(-1, 2).sum(axis=1)

        influence = -SCALE * (influence_dim1 - influence_dim2)
        top = np.argsort(influence)
        if beta_diff < 0:
            top = top[::-1]

        change = np.sum(influence[top[:alphaN]])
        new_beta_diff_amip = beta_diff + change
        change_sign_amip = np.sign(new_beta_diff_amip) != np.sign(beta_diff)

        if refit:
            res = run_logistic_regression(
                nonWeightedX[top[alphaN:],], nonWeightedY[top[alphaN:]],
                fit_intercept=self.fit_intercept,
                penalty=self.penalty
            )
            new_beta_diff_refit = SCALE * (
                res.coef_[0][dim_1] - res.coef_[0][dim_2]
            )
            change_sign_refit = np.sign(new_beta_diff_refit) != np.sign(beta_diff)
        else:
            new_beta_diff_refit = None
            change_sign_refit = None

        return change_sign_amip, change_sign_refit, beta_diff, new_beta_diff_amip, new_beta_diff_refit, top[:alphaN]


    def get_model(self):
        return self.model

    def get_pos_p_hats(self):
        return self.pos_p_hats



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

