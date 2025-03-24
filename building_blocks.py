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


def compute_approx(
    pos_p_hats: np.ndarray,
    X: np.ndarray,
    index: int,
    y: np.ndarray,
    method: str,
) -> float:
    """
    pos_p_hats: np.array, shape (n,), the predicted probabilities.
    X: np.array, shape (n, p), the design matrix.
    index: int, the index of the data point whose influence we want to compute.
    y: np.array, shape (n,), the response variable.
    method: str, the method to use to compute the approximation ("1sN" or "IF").

    Compute the influence function approximation of
    the effect of infinitesimally upweighting the index-th
    data point on the logistic regression coefficient.
    """
    v_lst = pos_p_hats * (1 - pos_p_hats)
    V = np.diag(v_lst)
    if method == "IF":
        influence_function = (
            np.linalg.inv(X.T @ V @ X) * (y[index] - pos_p_hats[index]) * X[index][0]
        )
        return influence_function[0][0]
    elif method == "1sN":
        influence_function = (
            np.linalg.inv(X.T @ V @ X) @ X[index] * (y[index] - pos_p_hats[index])
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


def plot_data(
    X: np.ndarray,
    y: np.ndarray,
    model: LogisticRegression,
    fit_range: int,
    filename: str,
    show_or_save: str,
) -> float:
    """
    X: np.array, shape (n, p), the design matrix.
    y: np.array, shape (n,), the response variable.
    model: a logistic regression model.
    fit_range: int, amount to extend the range of the fitted logistic regression
    beyond the data range.
    filename: str, the name of the file to save the plot to.
    show_or_save: str, whether to show or save the plot. "show" or "save".

    Create a scatter plot of the data with model fit overlayed.
    """
    # Create a range of values for plotting the decision boundary
    X_range = np.linspace(min(X) - fit_range, max(X) + fit_range, 100).reshape(-1, 1)
    y_prob = model.predict_proba(X_range)[:, 1]
    # Plotting
    plt.figure(figsize=(5, 2.5))
    plt.scatter(X, y, c=y, alpha=0.5, label="Original Data")
    plt.plot(X_range, y_prob, color="red", label="Logistic Regression Fit")
    plt.xlabel("Feature")
    plt.ylabel("Y")
    plt.axhline(0.5, color="grey", linestyle="--", label="Decision Boundary (y=0.5)")
    if show_or_save == "show":
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    pass
