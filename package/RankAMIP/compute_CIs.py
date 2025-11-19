# Construct Bootstrap Confidence Intervals (following from the Chatbot Arena Colab Notebook)
# https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=K9Plp9KhAu2n
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def compute_mle_bt(df):

    rawBT = df[['model_a', 'model_b', 'winner_model_a', 'winner_tie']]

    # make weighted design matrix for BT.
    X, y, player_to_id = make_BT_design_matrix(rawBT, weight_tie = True)

    model_full = run_logistic_regression(X, y)
    # prepend model 0, the reference model.
    bt_scores = np.insert(model_full.coef_[0], 0, 0)

    # combine bt_scores with player names
    id_to_player = {v: k for k, v in player_to_id.items()}
    return pd.Series(bt_scores, index=id_to_player.values()).sort_values(ascending=False)

def compute_mle_elo(
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df

def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


# BOOTSTRAP_ROUNDS = 100
# ### Obtain Bootstrap Confidence Intervals.
# np.random.seed(42)
# bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, BOOTSTRAP_ROUNDS)

# bootstrap_elo_lu.describe()
# # show the 95% confidence interval for each model
# ci_lower = bootstrap_elo_lu.quantile(0.025)
# ci_upper = bootstrap_elo_lu.quantile(0.975)
# ci_med = bootstrap_elo_lu.median()
# ci_df = pd.DataFrame({
#     "2.5%": ci_lower,
#     "Median": ci_med,
#     "97.5%": ci_upper
# })

# given ci_df, plot error bars for each model
def plot_elo_confidence_intervals(ci_df, title="Elo Ratings with 95% Confidence Intervals"):
    bars = pd.DataFrame(dict(
        model_name = ci_df.index,
        lower = ci_df["2.5%"],
        rating = ci_df["Median"],
        upper = ci_df["97.5%"]
    )).reset_index().sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 2)

    # Set global font to monospace and increase default font size
    plt.rcParams.update({
        'font.family': 'monospace',
        'font.size': 14
    })

    # Plot
    plt.figure(figsize=(12, 9), dpi=250)

    # Plot dots with error bars instead of horizontal bars
    plt.errorbar(bars['rating'], range(len(bars)),
                xerr=[bars['error_y_minus'], bars['error_y']],
                fmt='o', color="#f38d21", ecolor='black',
                capsize=5, markersize=10)  # Increased from 6 to 10

    # Add rating labels
    for index, row in bars.iterrows():
        plt.text(row['rating'] + row['error_y'] + 5, index, str(row['rating_rounded']), va='center', fontsize=20)

    # Set y-axis labels to model names with larger font
    plt.yticks(range(len(bars)), bars['model_name'], fontsize=14)
    plt.xticks(fontsize=14)

    plt.xlabel('Elo Rating', fontsize=16)
    # plt.title(title, fontsize=18)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.4)

    # Remove top and right spines (lines)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()
    plt.close()