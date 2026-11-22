"""Predicts the experience (appreciation, motivation, concentration) of each level
from a yoked agency model fitted on the players' sessions.

Pipeline, for one session_id:
  1. Load <StoredData>/<session_id>_trajectories.csv and cut it into actions
     (one action = one constant arrow-key command, see trajectories_to_actions).
  2. Yoked run (as in FunModelling/src/experiments/yoked_n_actions_n_targets.py):
     a D -> M agent passively receives the player's commands and their outcomes,
     and reports its sense of agency after each action (the 7 metrics).
     The signals are cached in <StoredData>/LevelPredictions/<session_id>_agency_signals.csv.
  3. Load the questionnaire answers <StoredData>/<session_id>_answers.csv (Likert 1-6).
  4. Fit "agency signals -> final answers" as in FunModelling/fit_time_series.py
     (per-session summary of each metric, then Ridge), on every session of StoredData
     that has both trajectories and answers (one session = one data point).
  5. Apply the fitted parameters to the agency signals of each level listed in
     level_to_experience.csv, and save the predictions.

Usage:
    python predict_level_experience.py <session_id> [--stored-data DIR] [--recompute]
or from Unity (PythonRunner): JSON on stdin, e.g.
    {"session_id": "20261122_114019", "stored_data_dir": ".../StoredData"}
A JSON summary is printed on stdout; progress messages go to stderr.

Outputs (in <StoredData>/LevelPredictions/):
    <session_id>_agency_signals.csv     yoked agent's signals after each of the player's actions
    <session_id>_fit.json               fitted parameters (scaler + ridge, per affect)
    <session_id>_level_predictions.csv  predicted affects per level
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FUN_MODELLING_DIR = HERE / "FunModelling"
sys.path.insert(0, str(FUN_MODELLING_DIR))  # FunModelling is not a package: imports are relative to its root

import fit_time_series as fts  # noqa: E402  (METRICS, AFFECTS, summarize, cv_r2, ...)
import src.utils.common as ucommon  # noqa: E402
from src.model.base import GoalPerceptionModel, MaxGoalPerceptionModel  # noqa: E402
from src.model.block_agency import BlockAgency, ChainBlockAgency  # noqa: E402
from src.model.forward_model import GaussianProcessForwardModel  # noqa: E402

# =============================================================================
# CONFIG
# =============================================================================
DEFAULT_STORED_DATA = HERE.parents[2] / "StoredData"          # <project>/StoredData
LEVEL_CSV = HERE.parent / "level_to_experience.csv"            # Assets/StreamingAssets/level_to_experience.csv
DUMMY_LEVEL_CSV = HERE / "level_to_experience_dummy.csv"       # used while LEVEL_CSV is empty

# Likert options of the questionnaire (Resources/*/GameQuestionnaire/options.txt) -> 1..6
LIKERT = {
    "fortement en désaccord": 1, "en désaccord": 2, "plutôt en désaccord": 3,
    "plutôt d'accord": 4, "d'accord": 5, "fortement d'accord": 6,
}
# question_index in <session_id>_answers.csv -> affect (order of the GameQuestionnaire)
QUESTION_TO_AFFECT = {0: "appreciation", 1: "motivation", 2: "concentration"}
QUESTIONNAIRE_ID = "GameQuestionnaire"

# Agent (same D -> M architecture as yoked_n_actions_n_targets.py, adapted to Unity)
COMMAND_BOUNDS = np.array([[-1., 1.],    # arrow keys: action_x, action_z in {-1, 0, 1}
                           [-1., 1.]])
POSITION_BOUNDS = np.array([[-10., 10.],   # D block's action = intended absolute position (x, z) on the board
                            [-10., 10.]])
FIND_ACTION_KWARGS = dict(n_iters=10, n_bootstrap=10, n_mutations=10)

# Fit: below this number of sessions, the fit falls back to the mean answer (intercept only)
MIN_SESSIONS_FOR_FIT = 3
SUMMARY_STAT = None  # None = all fts.SUMMARY_STATS; or e.g. "mean" to use only per-session means


def log(*args):
    print(*args, file=sys.stderr)


# =============================================================================
# 1. TRAJECTORIES -> ACTIONS
# =============================================================================
def trajectories_to_actions(traj: pd.DataFrame) -> pd.DataFrame:
    """
    One action = a maximal stretch of rows with the same non-zero command (action_x, action_z).
    Its outcome runs from the row before it starts until the next non-zero command starts
    (so the coasting after the key is released belongs to the action that caused it).
    Rows with no command before the first action of an episode are ignored.

    Returns one row per action: episode, command_x, command_z, start/end position (x, z),
    displacement (dx, dz) and score gain (number of pickups collected during the action).
    """
    actions = []
    for episode, g in traj.sort_values(["episode_id", "step_id"]).groupby("episode_id", sort=True):
        g = g.reset_index(drop=True)
        cmd = g[["action_x", "action_z"]].to_numpy()
        active = np.any(cmd != 0, axis=1)
        # start of a new command: active row whose command differs from the previous row's
        prev = np.vstack([np.zeros((1, 2)), cmd[:-1]])
        starts = np.where(active & np.any(cmd != prev, axis=1))[0]
        for k, s in enumerate(starts):
            e = starts[k + 1] - 1 if k + 1 < len(starts) else len(g) - 1   # last row of the outcome
            s0 = max(s - 1, 0)                                              # position when the key was pressed
            start = g.loc[s0, ["pos_x", "pos_z"]].to_numpy(float)
            end = g.loc[e, ["pos_x", "pos_z"]].to_numpy(float)
            actions.append({
                "episode": episode,
                "command_x": cmd[s, 0], "command_z": cmd[s, 1],
                "start_x": start[0], "start_z": start[1],
                "end_x": end[0], "end_z": end[1],
                "dx": end[0] - start[0], "dz": end[1] - start[1],
                "score_gain": float(g.loc[e, "score"] - g.loc[s0, "score"]),
                "t_start": g.loc[s, "time"], "t_end": g.loc[e, "time"],
            })
    return pd.DataFrame(actions)


# =============================================================================
# 2. YOKED AGENT
# Observation given to the agent after each action: [score_gain, end_x, end_z]
# - D block: picks a target position (absolute, x z), predicts the score gain (maximised)
# - M block: picks a 2D command (the Unity arrow keys), predicts the final position
#   (goal = D's target position)
# =============================================================================
def _D_states_to_goal(states):
    return np.asarray(states)[..., 0]      # score gain


def _M_states_to_goal(states):
    return np.asarray(states)[..., 1:3]    # final position (x, z)


def build_agent() -> ChainBlockAgency:
    gpm_D = MaxGoalPerceptionModel()
    gpm_D.translate_states_to_goal = _D_states_to_goal
    gpm_D.translate_predictions_to_goal = lambda x: x
    gpm_D.translate_states_to_predictions = _D_states_to_goal
    D_block = BlockAgency(goal=-np.array([np.nan]), action_bounds=POSITION_BOUNDS,
                          forward_model=GaussianProcessForwardModel(out_dim=1),
                          goal_perception_model=gpm_D, name="D_block")

    gpm_M = GoalPerceptionModel()
    gpm_M.translate_states_to_goal = _M_states_to_goal
    gpm_M.translate_predictions_to_goal = lambda x: x
    gpm_M.translate_states_to_predictions = _M_states_to_goal
    M_block = BlockAgency(goal=np.zeros(2), action_bounds=COMMAND_BOUNDS,
                          forward_model=GaussianProcessForwardModel(out_dim=2),
                          goal_perception_model=gpm_M, name="M_block")

    return ChainBlockAgency(blocks=[D_block, M_block], start_block_id=0, end_block_id=1,
                            transitions=np.array([[0, 1], [0, 0]]), blocks_attention=[1, 1])


# Same as yoked_anticipate / yoked_update in yoked_n_actions_n_targets.py
# (that module runs its whole experiment on import, so they cannot be imported).
def yoked_anticipate(agent, obs, observed_action):
    D_block, M_block = agent.blocks
    own_action = agent.find_action(obs, **FIND_ACTION_KWARGS)          # own decision, not executed
    own_target = agent.stored_actions[0]
    expected_target = M_block.forward_model(observed_action[np.newaxis, :])[0]
    M_block.predictions[-1] = expected_target                          # expectation of the other's action
    D_block.predictions[-1] = D_block.forward_model(expected_target[np.newaxis, :])[0]
    return own_target, own_action, expected_target


def yoked_update(agent, observed_action, obs):
    agent.stored_actions = {0: _M_states_to_goal(obs), 1: observed_action}  # "they aimed where they landed"
    agent.update(observed_action, obs)


def run_yoked(actions: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Yoked agent's sense of agency after each of the player's actions (one row per action)."""
    np.random.seed(seed)
    agent = build_agent()
    obs = np.zeros(3)
    rows = []
    for i, a in actions.iterrows():
        command = np.array([a.command_x, a.command_z], dtype=float)
        own_target, own_action, expected = yoked_anticipate(agent, obs, command)
        obs = np.array([a.score_gain, a.end_x, a.end_z])
        yoked_update(agent, command, obs)

        soa = ucommon.flatten(agent.report_sense_of_agency(), keep_parent_key=False)
        row = {"action": i, "episode": a.episode, "reward": a.score_gain}
        row.update({k: float(v) if v is not None else np.nan for k, v in soa.items()})
        row.update({"own_target_0": own_target[0], "own_target_1": own_target[1],
                    "own_action_0": own_action[0], "own_action_1": own_action[1],
                    "expected_target_0": expected[0], "expected_target_1": expected[1]})
        rows.append(row)
        log(f"  action {i + 1}/{len(actions)}")
    return pd.DataFrame(rows)


def agency_signals(session_id: str, stored: Path, out_dir: Path, recompute: bool = False) -> pd.DataFrame | None:
    """Cached yoked signals of a session; None if it has no usable trajectory."""
    cache = out_dir / f"{session_id}_agency_signals.csv"
    if cache.exists() and not recompute:
        return pd.read_csv(cache)
    traj_path = stored / f"{session_id}_trajectories.csv"
    if not traj_path.exists():
        return None
    actions = trajectories_to_actions(pd.read_csv(traj_path))
    if actions.empty:
        return None
    log(f"[{session_id}] yoked run on {len(actions)} actions")
    signals = run_yoked(actions)
    signals = pd.concat([signals, actions.drop(columns="episode")], axis=1)
    signals.to_csv(cache, index=False)
    return signals


# =============================================================================
# 3. ANSWERS
# =============================================================================
def load_answers(session_id: str, stored: Path) -> pd.Series | None:
    """Final affects of a session (Likert 1-6), indexed by affect; None if missing."""
    path = stored / f"{session_id}_answers.csv"
    if not path.exists():
        return None
    ans = pd.read_csv(path)
    ans = ans[ans["questionnaire_id"] == QUESTIONNAIRE_ID]
    ans = ans.drop_duplicates("question_index", keep="last")   # if answered twice, keep the last one
    out = {}
    for _, r in ans.iterrows():
        affect = QUESTION_TO_AFFECT.get(int(r["question_index"]))
        value = LIKERT.get(str(r["answer"]).strip())
        if affect is not None and value is not None:
            out[affect] = value
    if set(out) != set(fts.AFFECTS):
        log(f"[{session_id}] incomplete answers {out}: session ignored")
        return None
    return pd.Series(out)[fts.AFFECTS]


# =============================================================================
# 4. FIT (as in fit_time_series.py: per-session summary of the metrics -> Ridge)
# =============================================================================
def summarize_signals(signals_by_group: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per group (session or level), columns '<metric>_<stat>'."""
    long = pd.concat([s.assign(**{fts.PLAYER_COL: g}).rename(columns={"action": fts.TIME_COL})
                      for g, s in signals_by_group.items()], ignore_index=True)
    long[fts.METRICS] = long[fts.METRICS].astype(float).fillna(0.0)
    summary = fts.summarize(long.sort_values([fts.PLAYER_COL, fts.TIME_COL]), fts.METRICS)
    if SUMMARY_STAT is not None:
        summary = summary[[f"{m}_{SUMMARY_STAT}" for m in fts.METRICS]]
    return summary


def fit(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    """Fitted parameters per affect, as plain numbers (prediction = intercept + coef . (x - mean) / scale)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    params = {"features": list(X.columns), "n_sessions": len(X), "sessions": list(X.index)}
    if len(X) < MIN_SESSIONS_FOR_FIT:
        params["mode"] = "intercept_only"
        params["note"] = (f"only {len(X)} session(s) with trajectories and answers "
                          f"(< {MIN_SESSIONS_FOR_FIT}): predicting the mean answer")
        params["affects"] = {a: {"intercept": float(Y[a].mean()), "coef": [0.0] * X.shape[1]}
                             for a in fts.AFFECTS}
        params["scaler"] = {"mean": [0.0] * X.shape[1], "scale": [1.0] * X.shape[1]}
        return params

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    params["mode"] = "ridge"
    params["scaler"] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    params["affects"] = {}
    for a in fts.AFFECTS:
        ridge = RidgeCV(alphas=fts.RIDGE_ALPHAS).fit(Xs, Y[a])
        params["affects"][a] = {"intercept": float(ridge.intercept_), "coef": ridge.coef_.tolist(),
                                "alpha": float(ridge.alpha_)}
        if len(X) >= 2 * fts.CV_FOLDS:  # only meaningful with enough sessions
            params["affects"][a]["cv_r2"] = float(fts.cv_r2(X, Y[a]))
    return params


def predict(params: dict, X: pd.DataFrame) -> pd.DataFrame:
    X = X.reindex(columns=params["features"]).fillna(0.0).to_numpy(float)
    Xs = (X - np.array(params["scaler"]["mean"])) / np.array(params["scaler"]["scale"])
    out = {a: np.clip(p["intercept"] + Xs @ np.array(p["coef"]), *fts.LIKERT_RANGE)
           for a, p in params["affects"].items()}
    return pd.DataFrame(out)


# =============================================================================
# 5. LEVELS
# level_to_experience.csv: one row per (level, action), with the 7 metrics as columns
# (agency signals expected in that level, e.g. from simulated players).
# =============================================================================
def load_levels() -> tuple[pd.DataFrame, Path]:
    path = LEVEL_CSV if LEVEL_CSV.exists() and LEVEL_CSV.stat().st_size > 0 else DUMMY_LEVEL_CSV
    if path == DUMMY_LEVEL_CSV:
        log(f"{LEVEL_CSV.name} is empty: using the DUMMY levels of {DUMMY_LEVEL_CSV.name}")
    levels = pd.read_csv(path)
    missing = [c for c in ["level", "action", *fts.METRICS] if c not in levels.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return levels, path


def main(session_id: str, stored: Path, recompute: bool = False) -> dict:
    out_dir = stored / "LevelPredictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # the requested session must be usable
    if load_answers(session_id, stored) is None:
        raise FileNotFoundError(f"No complete answers for session {session_id} in {stored}")
    if agency_signals(session_id, stored, out_dir, recompute) is None:
        raise FileNotFoundError(f"No usable trajectories for session {session_id} in {stored}")

    # training set: every session with answers and trajectories
    signals, answers = {}, {}
    for f in sorted(stored.glob("*_answers.csv")):
        sid = f.name[: -len("_answers.csv")]
        ans = load_answers(sid, stored)
        sig = agency_signals(sid, stored, out_dir, recompute and sid == session_id) if ans is not None else None
        if sig is not None:
            signals[sid], answers[sid] = sig, ans
    X = summarize_signals(signals)
    Y = pd.DataFrame(answers).T.loc[X.index]

    params = fit(X, Y)
    params["session_id"] = session_id
    fit_path = out_dir / f"{session_id}_fit.json"
    fit_path.write_text(json.dumps(params, indent=2))

    levels, level_path = load_levels()
    X_levels = summarize_signals({lvl: g for lvl, g in levels.groupby("level")})
    preds = predict(params, X_levels)
    preds.insert(0, "level", X_levels.index)
    preds["fit_mode"] = params["mode"]
    preds["n_sessions"] = params["n_sessions"]
    preds["level_source"] = level_path.name
    pred_path = out_dir / f"{session_id}_level_predictions.csv"
    preds.to_csv(pred_path, index=False)

    return {
        "ok": True,
        "session_id": session_id,
        "fit_mode": params["mode"],
        "n_sessions": params["n_sessions"],
        "signals_path": str(out_dir / f"{session_id}_agency_signals.csv"),
        "fit_path": str(fit_path),
        "predictions_path": str(pred_path),
        "predictions": preds[["level", *fts.AFFECTS]].to_dict(orient="records"),
    }


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("session_id", nargs="?", help="if omitted, read JSON from stdin")
        parser.add_argument("--stored-data", default=None, help=f"default: {DEFAULT_STORED_DATA}")
        parser.add_argument("--recompute", action="store_true", help="redo the yoked run of this session")
        args = parser.parse_args()

        if args.session_id is None:   # called from Unity
            data = json.loads(sys.stdin.buffer.read().decode("utf-8-sig") or "{}")
            session_id = data["session_id"]
            stored = data.get("stored_data_dir") or args.stored_data or DEFAULT_STORED_DATA
            recompute = bool(data.get("recompute", args.recompute))
        else:
            session_id, stored, recompute = args.session_id, args.stored_data or DEFAULT_STORED_DATA, args.recompute

        print(json.dumps(main(session_id, Path(stored), recompute), default=float))
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
