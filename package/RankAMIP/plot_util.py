import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

def plot_data(
    X: np.ndarray,
    y: np.ndarray,
    model: LogisticRegression,
    fit_range: int,
    filename: str,
    show_or_save: str,
) -> None:
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