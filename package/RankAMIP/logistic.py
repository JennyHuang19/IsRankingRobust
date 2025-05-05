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

def isRankingRobust(k, alphaN, X, y):
    '''
    Check if the rank of the top k players/models is robust to data-dropping.
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
        sign_change_amip, sign_change_refit, original_beta_diff,new_beta_diff_amip, new_beta_diff_refit, indices = myAMIP.AMIP_sign_change(alphaN, playerA, playerB)
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
        

        self.__v__ = self.pos_p_hats * (1 - self.pos_p_hats)            # (n,)
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


    def AMIP_sign_change(self, alphaN, dim_1, dim_2 = None, 
                         method = "1sN", refit = True):
        '''
        AMIP to detect sign change of a parameter or difference between two parameters
        Arg: alphaN: int amount of data willing to drop
            dim_1, int, first parameter
            dim_2, int, second parameter, if not None, approximate the different between dim_1 and dim_2

        Return:
            change_sign_amip: bool, if amip says there is a sign change
            change_sign_refit: bool, if refit says there is a sign change
            new_beta_diff_amip: predicted new beta, or beta differences by AMIP
            new_beta_diff_refit: new beta or beta differences by refitting
            index
        '''
        if method == "1sN":
            get_influence = self.get_influence_1sN
        elif method == "IF":
            get_influence = self.get_influence_IF
        else:
            raise("method has to be 1sN or IF")
        beta = self.model.coef_[0]
    
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
                res = run_logistic_regression(self.X[top[alphaN:,]], 
                                              self.y[top[alphaN:]],
                                              fit_intercept=self.fit_intercept, 
                                              penalty=self.penalty
                                              )
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
            res = run_logistic_regression(self.X[top[alphaN:,]], 
                                          self.y[top[alphaN:]],
                                          fit_intercept=self.fit_intercept, 
                                          penalty=self.penalty)
            new_beta_diff_refit = res.coef_[0][dim_1] - res.coef_[0][dim_2]
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

