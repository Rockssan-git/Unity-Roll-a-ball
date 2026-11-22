"""
Time-series analysis of players' affects during a game,
compared with the agency model's estimates (the 7 game metrics).

Data expected (REAL DATA ONLY; to try without data run affect_timeseries_demo.py):

1. ACTIONS_CSV: one row per action
   id_joueur, action, <7 metrics>, appreciation, motivation, concentration
   (affects = 3 questions after each action, Likert 1-6)

2. FINAL_CSV: one row per player = final affects (the "global judgement"),
   answered by the player at the end of the game
   id_joueur, appreciation, motivation, concentration
   If FINAL_CSV is None, the last rating of each player is used instead.

Steps:
  1. Summarize each player's series -> one row per player
     - classic stats (mean, std, min, max, last, slope)
     - positive / negative split around a NEUTRAL value
       (positive peak, negative peak, end value, mean of positive part, mean of negative part)
  2. Global judgement = final affects
  3. Compare hypotheses: which summary of the game best predicts the final judgement?
     (peak-end rule with positive peak, negative peak, both, positive/negative means, ...)
  4. Do the agency model's metrics predict the final judgement?
  5. Compare summaries: player's affects vs model's metrics (correlations)
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIG
# =============================================================================
ACTIONS_CSV = None       # e.g. "data/actions.csv"
FINAL_CSV = None         # e.g. "data/final.csv"; None = use each player's last rating

PLAYER_COL = "id_joueur"
TIME_COL = "action"      # order of actions within a game (1, 2, 3, ...)
AFFECTS = ["appreciation", "motivation", "concentration"]
METRICS = [
    "reward", "convergence_selected_action", "goal_closure", "competence_progress",
    "prediction_error", "learning_progress", "feeling_of_effort",
]
LIKERT_RANGE = (1, 6)

# Neutral value: ratings above = positive moments, below = negative moments.
# A 1-6 scale has no middle answer, so the default is the midpoint 3.5. Change if needed.
NEUTRAL = 3.5

# Classic statistics used to summarize each series (add or remove freely)
SUMMARY_STATS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "last": lambda s: s.iloc[-1],
    "slope": lambda s: np.polyfit(np.arange(len(s)), s, 1)[0] if len(s) > 1 else 0.0,  # trend over the game
}

# Hypotheses to compare: which summary of the game explains the final judgement?
# Each final affect is predicted from the SAME affect's series
# (e.g. final appreciation from the appreciation ratings during the game).
# Available features: any key of SUMMARY_STATS, plus
#   peak_pos  = highest point above neutral (0 if never positive)
#   peak_neg  = lowest point below neutral  (0 if never negative)
#   end       = last rating, relative to neutral
#   mean_pos  = average of the positive moments only (0 if none)
#   mean_neg  = average of the negative moments only (0 if none)
HYPOTHESES = {
    "peak-end (positive peak)": ["peak_pos", "end"],
    "peak-end (negative peak)": ["peak_neg", "end"],
    "peak-end (both peaks)":    ["peak_pos", "peak_neg", "end"],
    "positive/negative means":  ["mean_pos", "mean_neg"],
    "plain average":            ["mean"],
    "all classic stats":        list(SUMMARY_STATS),
}

RIDGE_ALPHAS = np.logspace(-2, 3, 20)   # candidate penalty strengths; the best one is picked automatically
CV_FOLDS = 5

# --- Optional: real-time fun slider (Likert 1-6, dominant hand) ---------------
# If the slider is used, add a column "fun_slider" to ACTIONS_CSV
# (e.g. the slider value averaged between two actions) and uncomment:
# AFFECTS = AFFECTS + ["fun_slider"]


# =============================================================================
# STEP 0 - LOAD DATA
# =============================================================================
def load(actions_csv=ACTIONS_CSV, final_csv=FINAL_CSV):
    if actions_csv is None:
        sys.exit("No data: set ACTIONS_CSV in the CONFIG section (or run affect_timeseries_demo.py).")
    actions = pd.read_csv(actions_csv).sort_values([PLAYER_COL, TIME_COL])
    missing = [c for c in [PLAYER_COL, TIME_COL, *AFFECTS, *METRICS] if c not in actions.columns]
    if missing:
        sys.exit(f"Missing columns in {actions_csv}: {missing}")
    if final_csv is None:
        final = actions.groupby(PLAYER_COL)[AFFECTS].last()
    else:
        final = pd.read_csv(final_csv).set_index(PLAYER_COL)[AFFECTS]
    return actions, final


# =============================================================================
# STEP 1 - SUMMARIZE EACH PLAYER'S SERIES
# =============================================================================
def valence_features(s):
    """Positive / negative split of one series around NEUTRAL."""
    d = s.values - NEUTRAL
    pos, neg = d[d > 0], d[d < 0]
    return {
        "peak_pos": pos.max() if len(pos) else 0.0,
        "peak_neg": neg.min() if len(neg) else 0.0,
        "end": d[-1],
        "mean_pos": pos.mean() if len(pos) else 0.0,
        "mean_neg": neg.mean() if len(neg) else 0.0,
    }


def summarize(actions, columns, with_valence=False):
    """One row per player, one column per (variable, feature), e.g. 'motivation_peak_pos'."""
    rows = {}
    for player, g in actions.groupby(PLAYER_COL):
        row = {}
        for c in columns:
            row.update({f"{c}_{name}": float(f(g[c])) for name, f in SUMMARY_STATS.items()})
            if with_valence:
                row.update({f"{c}_{name}": float(v) for name, v in valence_features(g[c]).items()})
        rows[player] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# =============================================================================
# STEPS 3-4 - WHAT PREDICTS THE FINAL JUDGEMENT?
# Ridge = linear regression with a penalty that stops it from overfitting.
# R² is computed on players left out of training (1 = perfect, 0 = useless,
# below 0 = worse than always guessing the average).
# =============================================================================
def cv_r2(X, y):
    cv = KFold(min(CV_FOLDS, len(X)), shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    return cross_val_score(model, X, y, cv=cv, scoring="r2").mean()


def compare_hypotheses(affect_summary, final):
    print("\n--- Which summary of the game best predicts the final judgement? (cross-validated R²) ---")
    table = pd.DataFrame(index=list(HYPOTHESES), columns=AFFECTS, dtype=float)
    for name, feats in HYPOTHESES.items():
        for a in AFFECTS:
            X = affect_summary.loc[final.index, [f"{a}_{f}" for f in feats]]
            table.loc[name, a] = cv_r2(X, final[a])
    print(table.round(2))


def agency_model_power(metric_summary, final):
    print("\n--- Final judgement predicted from the agency model's metrics (cross-validated R²) ---")
    for a in AFFECTS:
        print(f"{a:15s} R² = {cv_r2(metric_summary.loc[final.index], final[a]):.2f}")


# =============================================================================
# STEP 5 - COMPARE SUMMARIES: PLAYER'S AFFECTS vs MODEL'S METRICS
# Spearman correlation (rank-based, suited to Likert answers), on per-player means.
# =============================================================================
def compare_summaries(affect_summary, metric_summary, stat="mean"):
    print(f"\n--- Correlation between player affects and model metrics (per-player {stat}, Spearman) ---")
    table = pd.DataFrame(index=METRICS, columns=AFFECTS, dtype=float)
    pvals = table.copy()
    for a in AFFECTS:
        for m in METRICS:
            rho, p = spearmanr(affect_summary[f"{a}_{stat}"], metric_summary[f"{m}_{stat}"])
            table.loc[m, a], pvals.loc[m, a] = rho, p
    print("rho (-1 to 1):")
    print(table.round(2))
    print("p-values:")
    print(pvals.round(3))


def main(actions_csv=ACTIONS_CSV, final_csv=FINAL_CSV):
    actions, final = load(actions_csv, final_csv)
    affect_summary = summarize(actions, AFFECTS, with_valence=True)
    metric_summary = summarize(actions, METRICS)

    print(f"{actions[PLAYER_COL].nunique()} players, {len(actions)} actions")
    print("\n--- Global judgement = final affects (over players) ---")
    print(final.describe().loc[["mean", "std", "min", "max"]].round(2))

    compare_hypotheses(affect_summary, final)
    agency_model_power(metric_summary, final)
    compare_summaries(affect_summary, metric_summary)


if __name__ == "__main__":
    main()


# =============================================================================
# (a) MOMENT-BY-MOMENT COMPARISON (kept for later; uncomment to use)
# Predict the affects at each action from the 7 metrics at that action.
# Test players are never seen in training (GroupKFold by player).
# =============================================================================
# from sklearn.model_selection import GroupKFold
# from sklearn.preprocessing import PolynomialFeatures
#
# def moment_by_moment(actions):
#     print("\n--- (a) Affects at each action predicted from the 7 metrics (cross-validated R²) ---")
#     for affect in AFFECTS:
#         model = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
#                               RidgeCV(alphas=RIDGE_ALPHAS))
#         r2 = cross_val_score(model, actions[METRICS], actions[affect], groups=actions[PLAYER_COL],
#                              cv=GroupKFold(CV_FOLDS), scoring="r2")
#         print(f"{affect:15s} R² = {r2.mean():.2f} (± {r2.std():.2f})")
#
# moment_by_moment(load()[0])
