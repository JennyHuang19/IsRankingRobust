import numpy as np
import pandas as pd
from package.RankAMIP.logistic import run_logistic_regression

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

def return_rankings_list(X, y, results, k, alphaN, player_to_id):
    '''
    X: np.ndarray, the design matrix.
    y: np.ndarray, the response variable.
    results: dict, the results dictionary.
    k: int, robustness of rankings of the top k players.
    alphaN: int, the number of players to remove.
    player_to_id: dict, dictionary mapping of player names to ids.

    return: nested list of players, where the inner list contains
    (player_name, playerid, old_score, new_score).
    '''
    playerA, playerB, orig_out, new_out, indices = results[k, alphaN]
    model_full = run_logistic_regression(X, y)
    Xd = np.delete(X, indices, axis=0)
    yd = np.delete(y, indices, axis=0)
    model_d  = run_logistic_regression(Xd, yd)
    
    # prepend model 0, the reference model, which has score 0.
    orig_scores = np.insert(model_full.coef_[0], 0, 0)
    new_scores = np.insert(model_d.coef_[0], 0, 0)

    # get play_id, orig_score
    indexed_orig_scores = list(enumerate(orig_scores))
    indexed_new_scores = list(enumerate(new_scores)) 
    # get player_id, sorted_new_score 
    sorted_original_scores = sorted(indexed_orig_scores, key=lambda x: x[1], reverse=True)
    sorted_new_scores = sorted(indexed_new_scores, key=lambda x: x[1], reverse=True)
    # get player_id, old_score, new_score 
    model_ranking_pre_post_drop = [[idx, val_a, indexed_new_scores[idx][1]] for idx, val_a in sorted_original_scores]
    # reverse the dictionary (player_id, player_name)
    id_to_player = {v: k for k, v in player_to_id.items()}
    # get player_name, player_id, old_score, new_score
    [elem.insert(0, id_to_player[elem[0]]) for elem in model_ranking_pre_post_drop]
    return model_ranking_pre_post_drop


def plot_bt_scores(X, y, rankings, alphaN, topk, filename_to_save):
    """
    Plots BT scores before and after data removal.
    
    Args:
    X: np.ndarray, the design matrix.
    y: np.ndarray, the response variable.
    rankings: list of tuples, (model_name, full_score, old_score, new_score)
    alphaN: int, number of dropped matches
    topk: int, number of top models to display
    filename_to_save: str, path to save the figure
    """
    # Extract top-k entries
    # Sorted by old_scores (index 2) in descending.
    rankings = sorted(rankings[:topk], key=lambda x: x[2], reverse=False)
    model_names = [x[0] for x in rankings[:topk]]
    old_scores = [x[2] for x in rankings[:topk]]
    new_scores = [x[3] for x in rankings[:topk]]
    num_matches_total = len(y)
    y_plot = np.arange(len(rankings[:topk]))
    offset = 0.15
    # Plot.
    # Set global font to monospace and increase default font size
    plt.rcParams.update({
        'font.family': 'monospace',
        'font.size': 14
    })

    # Plot
    plt.figure(figsize=(10, 9), dpi=250)

    # Scatter
    plt.scatter(old_scores, y_plot - offset, label='BT Score Full Data', marker='o', color='blue', s=72)
    plt.scatter(new_scores, y_plot + offset, 
                label=f'BT Score After Dropping {alphaN} out of {num_matches_total}\n'
                    f'({(alphaN/num_matches_total * 100):.3f}%) matches',
                marker='s', color='orange', s=72)

    # Extend x-axis limits slightly to the left and right
    min_score = min(min(old_scores), min(new_scores))
    max_score = max(max(old_scores), max(new_scores))
    plt.xlim(min_score - 0.05, max_score + 0.1)

    # Annotate scores next to points
    for i in range(len(y_plot)):
        plt.text(old_scores[i] + 0.01, y_plot[i] - offset, f'{old_scores[i]:.3f}', 
                va='center', ha='left', fontsize=14, fontfamily='monospace', color='blue')
        plt.text(new_scores[i] + 0.01, y_plot[i] + offset, f'{new_scores[i]:.3f}', 
                va='center', ha='left', fontsize=14, fontfamily='monospace', color='darkorange')

    # Axis
    plt.xlabel('Bradley-Terry Score', fontsize=16, fontfamily='monospace')
    plt.yticks(y_plot, model_names, fontsize=14, fontfamily='monospace')
    plt.xticks(fontsize=14, fontfamily='monospace')
    plt.title('Model Rankings in Chatbot Arena', fontsize=18, fontfamily='monospace')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save
    plt.savefig(filename_to_save, bbox_inches='tight')
    plt.close()