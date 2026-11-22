"""
Roll-a-Ball v2, standalone: ONE file. Put it at the root of the repository and run:
    python roll_a_ball_progress_v2.py

Change from v1: there is NO reward. Only internal signals guide the agent.
The D block chooses where to go by predicting, for each possible target, a WEIGHTED COMBINATION
of the agency signals that will follow:

    value = sum_i  w_i * z_i          z_i = signal i, standardised over the memory
    w_i   = SIGN_i * BASE_WEIGHT_i * relevance_i
    relevance_i = std of signal i over the last RELEVANCE_WINDOW steps / std over the whole memory

A signal that stopped changing (e.g. the score when nothing is ever collected) gets relevance ~0,
so its weight goes to 0 by itself (adaptive weights).

See README_roll_a_ball_v2.md for the list of changes and the choices to validate.
Only the original code of the repository is imported (src.utils, src.model).
"""
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

import src.utils.common as ucommon
import src.utils.display as udisplay
from src.model.base import GoalPerceptionModel, MaxGoalPerceptionModel
from src.model.block_agency import BlockAgency, ChainBlockAgency
from src.model.forward_model import (ForwardModel, GaussianProcessForwardModel,
                                     HeteroskedasticGaussianProcessRegressor)

# =============================================================================
# CONFIG
# =============================================================================
# --- Signals combined by the D block to choose its target ---
# SIGN: +1 = the agent seeks high values, -1 = low values, 0 = ignored.
# BASE_WEIGHT: weight before the adaptive relevance factor.
# These signs are CHOICES TO VALIDATE (see README).
SIGNALS = {
    #  name                         SIGN  BASE_WEIGHT
    "reward":                      (+1,  1.0),   # objectives collected during the step (score)
    "goal_closure":                (-1,  1.0),   # distance to the nearest objective (small = good)
    "competence_progress":         (-1,  1.0),   # change of goal_closure (negative = getting closer)
    "prediction_error":            (-1,  1.0),   # M block: error between predicted and actual move
    "learning_progress":           (-1,  1.0),   # M block: change of prediction error (negative = learning)
    "feeling_of_effort":           (-1,  1.0),   # M block: change of command between two steps
    "convergence_selected_action": (-1,  1.0),   # M block: change of best score at the end of the search
}
RELEVANCE_WINDOW = 6       # recent steps used to judge whether a signal still changes
PROGRESS_WINDOW = 4        # window of competence_progress (same as window_size of BlockAgency)

# --- Game (few trials, like the original experiments: 10 games x 4 actions = 40 trials) ---
SEED = 0
N_EPISODES = 10            # number of games
N_STEPS = 4                # max actions per game
OBJECTIVES = [[1, 1], [2, 2], [3, 3]]
ARENA = np.array([[-1., 4.], [-1., 4.]])   # walls: [low, high] per axis
COLLECT_RADIUS = 0.1
STEP_TIME = 2              # duration of one command (s)
COMMAND_BOUNDS = np.array([[-2., 2.], [-2., 2.]])
D_RADIUS = 1.0             # the D block picks a target within this distance of the ball
SEARCH = dict(n_iters=10, n_bootstrap=10, n_mutations=10)

# --- Gaussian processes ---
WR = 3.0                   # recency weight of the D block's GP: how fast old samples lose importance
MEMORY_SIZE = 200          # max samples kept; t_max = min(t + 1, MEMORY_SIZE)
USE_REMAINING_MASK = True  # add "which objectives are left" to the D block's input
N_RESTARTS = 2             # restarts of the GP optimiser (more = slower)
SIGMA0_SQ = 0.1            # base observation noise sigma_0² of the GPs (D and M), in units of the
                           # standardised outputs. None = fitted (original code: ends up ~0).

OUTPUT_FOLDER = "data/RollABall"
OUTPUT_NAME = "signals_v2"


# =============================================================================
# Environment (2D Roll-a-Ball: constant command against linear damping)
# =============================================================================
class RollABall:
    damping = 0.45

    def __init__(self):
        self.reset()

    def reset(self):
        self.objectives = np.array(OBJECTIVES, dtype=float)
        self.collected = np.zeros(len(self.objectives), dtype=bool)
        self.pos = np.zeros(2)
        self.last_disp = np.zeros(2)

    @property
    def done(self):
        return bool(self.collected.all())

    @property
    def mask(self):
        return (~self.collected).astype(float)

    def nearest_distance(self):
        left = self.objectives[~self.collected]
        return 0. if len(left) == 0 else np.linalg.norm(left - self.pos, axis=1).min()

    def step(self, command):
        """Returns (n collected, progress)."""
        nearest = np.flatnonzero(~self.collected)
        nearest = nearest[np.argmin(np.linalg.norm(self.objectives[nearest] - self.pos, axis=1))]
        d_before = np.linalg.norm(self.objectives[nearest] - self.pos)

        a, d = np.clip(command, COMMAND_BOUNDS[:, 0], COMMAND_BOUNDS[:, 1]), self.damping
        t = np.linspace(0, STEP_TIME, int(STEP_TIME * 24) + 1)[:, None]
        traj = np.clip(self.pos + a / d * t - a / d**2 * (1 - np.exp(-d * t)), ARENA[:, 0], ARENA[:, 1])

        hit = (cdist(traj, self.objectives).min(axis=0) < COLLECT_RADIUS) & ~self.collected
        self.collected |= hit
        self.last_disp = traj[-1] - self.pos
        self.pos = traj[-1]
        d_after = 0. if self.collected[nearest] else np.linalg.norm(self.objectives[nearest] - self.pos)
        return int(hit.sum()), d_before - d_after


# =============================================================================
# BlockAgency with fixes A1-A4 (see problemes_FunModelling.md); original file untouched
# =============================================================================
class FixedBlockAgency(BlockAgency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_closures = []

    def compute_competence_progress(self):  # A3: uses the closures recorded at each trial
        if len(self.goal_closures) < self.window_size:
            return 0
        recent = np.asarray(self.goal_closures[-self.window_size:], dtype=float)
        w = self.window_size // 2 + self.window_size % 2
        return np.mean(recent[-w:]) - np.mean(recent[:w])

    def update(self, action, obs):
        self.goal_closure = self.goal_perception_model.distance_states_to_goal(obs, self.goal)
        self.goal_closures.append(float(np.squeeze(self.goal_closure)))
        self.errors.append(np.linalg.norm(
            self.goal_perception_model.translate_states_to_predictions(obs) - self.predictions[-1]))
        self.prediction_error = self.errors[-1]
        self.states.append(obs)
        self.competence_progress = self.compute_competence_progress()
        self.learning_progress = self.compute_learning_progress()
        self.feeling_of_effort = 0 if self.prev_action is None else np.linalg.norm(action - self.prev_action)
        self.prev_action = action
        self.update_sense_of_agency()
        self.forward_model.update(action, self.goal_perception_model.translate_states_to_predictions(obs))

    def mutate(self, actions, n_mutations, ratio_std=1 / 20):  # A4: noise centred on 0
        width = self.action_bounds[:, 1] - self.action_bounds[:, 0]
        noise = ratio_std * np.random.uniform(-width, width, (n_mutations, *actions.shape))
        return (actions[None] + noise).reshape(-1, self.action_dim)

    def find_action(self, obs, n_iters, n_bootstrap, n_mutations, top_ratio_kept=.1, min_delta=1e-3):
        candidates = self.random_sample(n_bootstrap)
        predictions = self.forward_model(candidates)
        scores = np.asarray(self.objective_func(predictions)).flatten()
        best_candidate, best_prediction, best_score = None, None, None
        for _ in range(n_iters):
            n_kept = max(1, int(len(scores) * top_ratio_kept))
            idx = np.argsort(scores)[:n_kept]
            kept_c = candidates[idx].reshape((-1, self.action_dim))
            kept_p = predictions[idx].reshape((-1, self.forward_model.out_dim))
            kept_s = scores[idx]
            best_candidate, best_prediction = kept_c[0], kept_p[0]          # A1
            if best_score is not None:                                       # A2
                self.convergence_selected_action = float(abs(best_score - kept_s[0]))
                if self.convergence_selected_action < min_delta:
                    break
            best_score = kept_s[0]
            new_c = self.mutate(kept_c, n_mutations)
            new_p = self.forward_model(new_c)
            candidates = np.concatenate([kept_c, new_c])
            predictions = np.concatenate([kept_p, new_p])
            scores = np.concatenate([kept_s, np.asarray(self.objective_func(new_p)).flatten()])
        if self.convergence_selected_action is None:
            self.convergence_selected_action = 0.0
        self.predictions.append(best_prediction)
        return best_candidate


# =============================================================================
# Forward models
# =============================================================================
class ContextualForwardModel(ForwardModel):
    """Input = [context, action] (context = ball position), so M knows where the ball starts."""
    def __init__(self, inner):
        self.inner, self.out_dim, self.context = inner, inner.out_dim, None

    def set_context(self, context):
        self.context = np.asarray(context, dtype=float)

    def __call__(self, actions):
        actions = np.atleast_2d(actions)
        return self.inner(np.hstack([np.tile(self.context, (len(actions), 1)), actions]))

    def update(self, action, obs):
        return self.inner.update(np.concatenate([self.context, action]), obs)




# =============================================================================
# D block's forward model: predicted combination of signals for a target
# =============================================================================
NAMES = list(SIGNALS)
SIGNS = np.array([SIGNALS[n][0] for n in NAMES], dtype=float)
BASE_WEIGHTS = np.array([SIGNALS[n][1] for n in NAMES], dtype=float)


class SignalCombinationModel(ForwardModel):
    """
    Stores, for each past decision, the target chosen (+ objectives left) and the RAW signals that
    followed. At each refit, the combination is recomputed with the CURRENT weights (so a signal whose
    weight drops to 0 stops counting in old memories too), then fitted by a heteroskedastic GP
    (recent samples count more, at most MEMORY_SIZE samples).
    __call__(actions) = predicted combination for targets ball position + action.
    """
    out_dim = 1

    def __init__(self):
        kernel_params = {"recency": {"recency_weight": WR, "recency_weight_bounds": "fixed", "t_max": 1.0}}
        if SIGMA0_SQ is not None:
            kernel_params["white"] = {"noise_level": SIGMA0_SQ, "noise_level_bounds": "fixed"}
        self.gp = HeteroskedasticGaussianProcessRegressor(
            kernel_params=kernel_params, n_restarts_optimizer=N_RESTARTS, normalize_y=True, random_state=SEED)
        self.scaler = StandardScaler()
        self.X, self.S, self.t = [], [], []     # inputs, raw signal vectors, global times
        self.step = 0
        self.pos, self.mask = None, None
        self.pending_signals = None

    # ---- inputs
    def features(self, pos, mask):
        pos = np.atleast_2d(pos)
        return np.hstack([pos, np.tile(mask, (len(pos), 1))]) if USE_REMAINING_MASK else pos

    def set_context(self, pos, mask):
        self.pos, self.mask = np.asarray(pos, float), np.asarray(mask, float)

    # ---- adaptive weights
    def weights(self):
        """w_i = SIGN_i * BASE_WEIGHT_i * relevance_i (relevance in [0, 1])."""
        if len(self.S) < 2:
            return SIGNS * BASE_WEIGHTS
        S = np.array(self.S)
        std_all = S.std(axis=0)
        std_recent = S[-RELEVANCE_WINDOW:].std(axis=0)
        relevance = np.divide(std_recent, std_all, out=np.zeros_like(std_all), where=std_all > 0)
        return SIGNS * BASE_WEIGHTS * np.clip(relevance, 0, 1)

    def combine(self, signals):
        """Weighted sum of the signals, each standardised over the memory."""
        signals = np.atleast_2d(signals)
        if len(self.S) < 2:
            return np.zeros(len(signals))
        S = np.array(self.S)
        mean, std = S.mean(axis=0), S.std(axis=0)
        z = np.divide(signals - mean, std, out=np.zeros_like(signals, dtype=float), where=std > 0)
        return z @ self.weights()

    # ---- prediction
    def __call__(self, actions):
        if not self.X:
            return np.zeros((len(np.atleast_2d(actions)), 1))
        X = self.features(self.pos + np.atleast_2d(actions), self.mask)
        t_max = min(self.t[-1] + 1, MEMORY_SIZE)
        return self.gp.predict(self.scaler.transform(X), t_current=self.t[-1] / t_max).reshape(-1, 1)

    # ---- learning
    def observe(self, signals):
        """Raw signals that followed the last decision (called just before the D block's update)."""
        self.pending_signals = np.asarray(signals, dtype=float)

    def update(self, action, obs):
        """Add (target, signals), keep the last MEMORY_SIZE, refit on the combination with current weights."""
        self.X.append(self.features(self.pos + np.asarray(action), self.mask)[0])
        self.S.append(self.pending_signals)
        self.t.append(self.step)
        self.X, self.S, self.t = self.X[-MEMORY_SIZE:], self.S[-MEMORY_SIZE:], self.t[-MEMORY_SIZE:]
        self.step += 1
        y = self.combine(np.array(self.S))
        t_max = min(self.t[-1] + 1, MEMORY_SIZE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.scaler.fit_transform(np.array(self.X)), y,
                        t=np.array(self.t) / t_max, t_current=self.t[-1] / t_max)


# =============================================================================
# Agent: D (where to go, maximises the predicted combination) -> M (which command)
# Observation given to the blocks: [ball position (2), last displacement (2), combination (1)]
# =============================================================================
def build_agent():
    sm = SignalCombinationModel()
    d_goal = MaxGoalPerceptionModel()
    value_of = lambda s: np.asarray(s)[..., -1]
    d_goal.translate_states_to_goal = value_of
    d_goal.translate_predictions_to_goal = lambda p: np.asarray(p)[..., 0]
    d_goal.translate_states_to_predictions = lambda s: value_of(s).reshape(-1)
    d_block = FixedBlockAgency(goal=np.array([np.nan]),
                               action_bounds=np.array([[-D_RADIUS, D_RADIUS], [-D_RADIUS, D_RADIUS]]),
                               forward_model=sm, goal_perception_model=d_goal, name="D_block")

    m_gp = GaussianProcessForwardModel(out_dim=2)
    if SIGMA0_SQ is not None:   # same kernel as the original, with the white noise fixed
        m_gp.gaussian_process.kernel.k2.noise_level = SIGMA0_SQ
        m_gp.gaussian_process.kernel.k2.noise_level_bounds = "fixed"
    m_fm = ContextualForwardModel(m_gp)
    m_goal = GoalPerceptionModel()
    disp_of = lambda s: np.asarray(s)[..., 2:4]
    m_goal.translate_states_to_goal = disp_of
    m_goal.translate_predictions_to_goal = lambda p: p
    m_goal.translate_states_to_predictions = disp_of
    m_block = FixedBlockAgency(goal=np.zeros(2), action_bounds=COMMAND_BOUNDS,
                               forward_model=m_fm, goal_perception_model=m_goal, name="M_block")

    agent = ChainBlockAgency(blocks=[d_block, m_block], start_block_id=0, end_block_id=1,
                             transitions=np.array([[0, 1], [0, 0]]), blocks_attention=[1, 1])
    return agent, sm, m_fm


def read_signals(env, collected, m_block, closures):
    """The signals of SIGNALS, after one step: game-level (score, distance) and M block (motor)."""
    closures.append(env.nearest_distance())
    if len(closures) < PROGRESS_WINDOW:
        progress = 0.
    else:
        recent = np.array(closures[-PROGRESS_WINDOW:])
        w = PROGRESS_WINDOW // 2 + PROGRESS_WINDOW % 2
        progress = recent[-w:].mean() - recent[:w].mean()
    values = {"reward": collected, "goal_closure": closures[-1], "competence_progress": progress,
              "prediction_error": m_block.prediction_error, "learning_progress": m_block.learning_progress,
              "feeling_of_effort": m_block.feeling_of_effort,
              "convergence_selected_action": m_block.convergence_selected_action}
    return np.array([float(values[n]) for n in NAMES])


def run(verbose=True):
    np.random.seed(SEED)
    env = RollABall()
    agent, sm, m_fm = build_agent()
    d_block, m_block = agent.blocks
    closures = []      # goal_closure history (for competence_progress)
    rows = []
    for episode in range(N_EPISODES):
        env.reset()
        for step in range(N_STEPS):
            obs = np.concatenate([env.pos, env.last_disp, [0.]])
            sm.set_context(env.pos, env.mask)
            start = env.pos.copy()
            d_action = d_block.find_action(obs, **SEARCH)

            m_block.goal = d_action                      # displacement wanted by D
            m_fm.set_context(start)
            command = m_block.find_action(obs, **SEARCH)
            collected, _ = env.step(command)
            m_block.update(command, np.concatenate([env.pos, env.last_disp, [0.]]))

            signals = read_signals(env, collected, m_block, closures)
            value = sm.combine(signals)[0]               # actual combination, with the current weights
            sm.observe(signals)
            d_block.update(d_action, np.concatenate([env.pos, env.last_disp, [value]]))

            agent.update_sense_of_agency()
            row = {"episode": episode, "step": step, "trial": len(rows), "collected": collected,
                   "objectives_left": int((~env.collected).sum()), "value": value,
                   "pos_x": env.pos[0], "pos_y": env.pos[1]}
            row.update({f"sig_{n}": v for n, v in zip(NAMES, signals)})
            row.update({f"w_{n}": v for n, v in zip(NAMES, sm.weights())})
            row.update(ucommon.flatten(agent.report_sense_of_agency(), keep_parent_key=False))
            rows.append(row)
            if env.done:
                break
        if verbose:
            print(f"episode {episode + 1}/{N_EPISODES}: collected {int(env.collected.sum())}/{len(env.collected)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = run()
    ucommon.recursive_mkdir(OUTPUT_FOLDER)
    data.to_csv(f"{OUTPUT_FOLDER}/{OUTPUT_NAME}.csv", index=False)
    # same plot as the original experiments (one row per curve, game boundaries in the background)
    udisplay.plot_values_across_trials(
        data, y_names=["collected", "value"] + [f"sig_{n}" for n in NAMES] + [f"w_{n}" for n in NAMES],
        sequence_groups="episode", sequence_style="background", sequence_labels=True,
        file_path=f"{OUTPUT_FOLDER}/{OUTPUT_NAME}.html")
    print(f"saved {OUTPUT_FOLDER}/{OUTPUT_NAME}.csv and .html")
