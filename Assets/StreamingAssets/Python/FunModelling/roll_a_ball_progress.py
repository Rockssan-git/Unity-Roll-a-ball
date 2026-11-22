"""
Roll-a-Ball, standalone version: ONE file, nothing else to copy.
Put it at the root of the repository and run:   python roll_a_ball_progress.py

Same agent as roll_a_ball_td.py (D block learns a value V(s) by TD, M block moves the ball),
with one addition: the reward used to learn V can include a "perceived progress" bonus.

    reward = objectives collected during the step + PROGRESS_WEIGHT * progress
    progress = how much closer the ball got to the nearest objective (distance before - after)

PROGRESS_WEIGHT = 0 -> only collected objectives count (sparse reward, like roll_a_ball_td.py).
PROGRESS_WEIGHT > 0 -> getting closer is rewarding too (like a player who sees the ball approach).

Only the original code of the repository is imported (src.utils, src.model); the environment,
the block fixes and the TD value model are all defined below.
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
PROGRESS_WEIGHT = 1     # weight of the "getting closer" bonus in the reward (0 = off)

SEED = 0
N_EPISODES = 30            # number of games
#N_STEPS = 6                # max actions per game
# OBJECTIVES = [[1, 1], [2, 2], [3, 3]]
# ARENA = np.array([[-1., 4.], [-1., 4.]])   # walls: [low, high] per axis
COLLECT_RADIUS = 0.1
OBJECTIVES = [[1, 1], [-1, 2], [3, -1], [2, 3], [-2, -2], [0, 3]]
ARENA = np.array([[-3., 4.], [-3., 4.]])
#N_STEPS = 10
N_STEPS = 16
STEP_TIME = 2              # duration of one command (s)
COMMAND_BOUNDS = np.array([[-2., 2.], [-2., 2.]])
D_RADIUS = 1.0             # the D block picks a target within this distance of the ball
SEARCH = dict(n_iters=10, n_bootstrap=10, n_mutations=10)

GAMMA = 0.95               # weight of future rewards
LR = 0.5                   # TD learning rate
WR = 3.0                   # recency weight: how fast old samples lose importance
MEMORY_SIZE = 200          # max samples kept by V
USE_REMAINING_MASK = True  # add "which objectives are left" to the state
N_RESTARTS = 2             # restarts of the GP optimiser (more = slower)

OUTPUT_FOLDER = "data/RollABall"
OUTPUT_NAME = f"hard_w{PROGRESS_WEIGHT}"


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


class TDValueModel(ForwardModel):
    """
    V(s) by TD(0) with a heteroskedastic GP: each TD target is added with its global time,
    older samples get more noise (WR).
    """
    out_dim = 1

    def __init__(self):
        self.gp = HeteroskedasticGaussianProcessRegressor(
            kernel_params={"recency": {"recency_weight": WR, "recency_weight_bounds": "fixed", "t_max": 1.0}},
            n_restarts_optimizer=N_RESTARTS, normalize_y=True, random_state=SEED)
        self.scaler = StandardScaler()
        self.X, self.y, self.t = [], [], []
        self.step = 0
        self.pos, self.mask = None, None

    def features(self, pos, mask):
        pos = np.atleast_2d(pos)
        return np.hstack([pos, np.tile(mask, (len(pos), 1))]) if USE_REMAINING_MASK else pos

    def set_context(self, pos, mask, objectives):
        """objectives: positions of all objectives (the player sees them)."""
        self.pos, self.mask = np.asarray(pos, float), np.asarray(mask, float)
        left = objectives[self.mask > 0]
        self.nearest = left[np.argmin(np.linalg.norm(left - self.pos, axis=1))]
        self.d_before = np.linalg.norm(self.nearest - self.pos)

    def value(self, X):
        if not self.X:
            return np.zeros(len(X))
        t_max = min(self.t[-1] + 1, MEMORY_SIZE)
        return self.gp.predict(self.scaler.transform(X), t_current=self.t[-1] / t_max).reshape(-1)

    def __call__(self, actions):
        """Predicted TD target of going to pos + action: perceived progress bonus + GAMMA * V(next state).
        (The bonus has to be counted here: V alone is HIGHER far from the objectives, because more
        progress is still to be made from there, so maximising V alone sends the ball away.)"""
        targets = self.pos + np.atleast_2d(actions)
        bonus = PROGRESS_WEIGHT * (self.d_before - np.linalg.norm(targets - self.nearest, axis=1))
        return (bonus + GAMMA * self.value(self.features(targets, self.mask))).reshape(-1, 1)

    def td_target(self, reward, next_pos, next_mask, terminal):
        v_next = 0. if terminal else self.value(self.features(next_pos, next_mask))[0]
        return reward + GAMMA * v_next

    def update(self, action, td_target):
        s = self.features(self.pos, self.mask)[0]
        v = self.value(s[None])[0]
        self.X.append(s); self.y.append(v + LR * (float(np.squeeze(td_target)) - v)); self.t.append(self.step)
        self.X, self.y, self.t = self.X[-MEMORY_SIZE:], self.y[-MEMORY_SIZE:], self.t[-MEMORY_SIZE:]
        self.step += 1
        t_max = min(self.t[-1] + 1, MEMORY_SIZE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp.fit(self.scaler.fit_transform(np.array(self.X)), np.array(self.y),
                        t=np.array(self.t) / t_max, t_current=self.t[-1] / t_max)


# =============================================================================
# Agent: D (where to go, maximises V) -> M (which command, reaches the target)
# Observation given to the blocks: [ball position (2), last displacement (2), TD target (1)]
# =============================================================================
def build_agent():
    vm = TDValueModel()
    d_goal = MaxGoalPerceptionModel()
    td_of = lambda s: np.asarray(s)[..., -1]
    d_goal.translate_states_to_goal = td_of
    d_goal.translate_predictions_to_goal = lambda p: np.asarray(p)[..., 0]
    d_goal.translate_states_to_predictions = lambda s: td_of(s).reshape(-1)
    d_block = FixedBlockAgency(goal=np.array([np.nan]),
                               action_bounds=np.array([[-D_RADIUS, D_RADIUS], [-D_RADIUS, D_RADIUS]]),
                               forward_model=vm, goal_perception_model=d_goal, name="D_block")

    m_fm = ContextualForwardModel(GaussianProcessForwardModel(out_dim=2))
    m_goal = GoalPerceptionModel()
    disp_of = lambda s: np.asarray(s)[..., 2:4]
    m_goal.translate_states_to_goal = disp_of
    m_goal.translate_predictions_to_goal = lambda p: p
    m_goal.translate_states_to_predictions = disp_of
    m_block = FixedBlockAgency(goal=np.zeros(2), action_bounds=COMMAND_BOUNDS,
                               forward_model=m_fm, goal_perception_model=m_goal, name="M_block")

    agent = ChainBlockAgency(blocks=[d_block, m_block], start_block_id=0, end_block_id=1,
                             transitions=np.array([[0, 1], [0, 0]]), blocks_attention=[1, 1])
    return agent, vm, m_fm


def run(verbose=True):
    np.random.seed(SEED)
    env = RollABall()
    agent, vm, m_fm = build_agent()
    d_block, m_block = agent.blocks
    rows = []
    for episode in range(N_EPISODES):
        env.reset()
        for step in range(N_STEPS):
            obs = np.concatenate([env.pos, env.last_disp, [0.]])
            vm.set_context(env.pos, env.mask, env.objectives)
            start = env.pos.copy()
            d_action = d_block.find_action(obs, **SEARCH)

            m_block.goal = d_action                      # displacement wanted by D
            m_fm.set_context(start)
            command = m_block.find_action(obs, **SEARCH)
            collected, progress = env.step(command)
            reward = collected + PROGRESS_WEIGHT * progress
            m_block.update(command, np.concatenate([env.pos, env.last_disp, [0.]]))

            terminal = env.done or step == N_STEPS - 1
            y_td = vm.td_target(reward, env.pos, env.mask, terminal)
            d_block.update(d_action, np.concatenate([env.pos, env.last_disp, [y_td]]))

            agent.update_sense_of_agency()
            row = {"episode": episode, "step": step, "trial": len(rows), "collected": collected,
                   "progress": progress, "reward": reward, "objectives_left": int((~env.collected).sum()),
                   "distance_to_nearest": env.nearest_distance(), "td_target": y_td,
                   "pos_x": env.pos[0], "pos_y": env.pos[1]}
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
    # same plot as the original experiments (one row per signal, game boundaries in the background)
    udisplay.plot_values_across_trials(
        data, y_names=["collected", "distance_to_nearest", "reward", "convergence_selected_action",
                       "goal_closure", "competence_progress", "prediction_error", "learning_progress",
                       "feeling_of_effort"],
        sequence_groups="episode", sequence_style="background", sequence_labels=True,
        file_path=f"{OUTPUT_FOLDER}/{OUTPUT_NAME}.html")
    print(f"saved {OUTPUT_FOLDER}/{OUTPUT_NAME}.csv and .html")
