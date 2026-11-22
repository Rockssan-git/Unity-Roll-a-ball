# common librairies
import numpy as np
import pandas as pd

# my scripts
import src.utils.common as ucommon
import src.utils.display as udisplay

from src.env.toy_envs import LinearDampingManyTargetsEnv
from src.model.block_agency import BlockAgency, ChainBlockAgency
from src.model.base import MaxGoalPerceptionModel, GoalPerceptionModel
from src.model.forward_model import GaussianProcessForwardModel


################################################################################
# Yoked condition
# Env = n actions, n targets
# Model block D = find next target position, block M = get to position
#
# A "master" agent acts in the env exactly as in n_actions_n_targets.py.
# A "yoked" agent, with the same architecture, never selects anything: it
# passively receives the master's commands and the resulting observations,
# learns from them and reports its own sense of agency.
#
# Each step, the yoked agent:
# 1. runs its own decision loop (ChainBlockAgency.find_action) on the current obs:
#    the D block forms its own intended target, sent as goal to the M block, which
#    selects its own command. This command is never executed, but the loop sets the
#    blocks' goals and convergence_selected_action (sense of initiation).
# 2. forms an expectation of the outcome of the *observed* (master's) command:
#    - M block: prediction = M.forward_model(observed command)
#    - D block: prediction = D.forward_model(expected final position)
#    These replace the predictions of its own (unexecuted) choice, so that:
#    goal_closure     = outcome vs its own intention
#    prediction_error = outcome vs its expectation of the other's action
# 3. learns from the observed command and outcome:
#    - M block: action = the observed command
#    - D block: action = the position actually reached (retrospective attribution:
#      "they aimed where they landed"), so the D forward model learns the true
#      position -> n_hits mapping.
################################################################################
command_bounds = np.array([[-2.,2.],
                           [-2.,2.]])
position_bounds = np.array([[-1.,4.],
                            [-1.,4.]]) # covers all env.goal_positions

# Env
env = LinearDampingManyTargetsEnv(
    action_bounds=command_bounds,
    several_steps=True
)
n_goals = len(env.goal_positions)

# (assumption: internalized score system)
def _D_translate_states_to_goal(states:np.ndarray):
    _states = states[...,2*n_goals:] # n goal_positions, then trajectory
    if len(_states.shape) < 2:
        _states = _states.reshape((-1,2))
    else:
        _states = _states.reshape((_states.shape[0],-1,2))
    n_hits = env.compute_reward(overide_states=_states)
    return n_hits


def _M_translate_states_to_goal(states:np.ndarray):
    return states[...,-2:] # only final 2D position of the trajectory


def build_agent()->ChainBlockAgency:
    """Same D -> M architecture as n_actions_n_targets.py, with fresh forward models."""
    # D block = decides next position
    goal_perception_model_D = MaxGoalPerceptionModel()
    goal_perception_model_D.translate_states_to_goal = _D_translate_states_to_goal
    goal_perception_model_D.translate_predictions_to_goal = lambda x:x
    goal_perception_model_D.translate_states_to_predictions = _D_translate_states_to_goal

    D_block_agency = BlockAgency(goal=-np.array([np.nan]), # not used with MaxGoalPerceptionModel
                                 action_bounds=position_bounds, # D block's action = position (sent as goal to M block)
                                 forward_model=GaussianProcessForwardModel(out_dim=1), # predict score
                                 goal_perception_model=goal_perception_model_D,
                                 name="D_block")

    # M block = decides 2D command from a 2D position
    goal_perception_model_M = GoalPerceptionModel()
    goal_perception_model_M.translate_states_to_goal = _M_translate_states_to_goal
    goal_perception_model_M.translate_predictions_to_goal = lambda x:x
    goal_perception_model_M.translate_states_to_predictions = _M_translate_states_to_goal

    M_block_agency = BlockAgency(goal=np.array([1.,1.]), # changed online from D block
                                 action_bounds=command_bounds, # M block's action = command
                                 forward_model=GaussianProcessForwardModel(out_dim=2), # predict final position
                                 goal_perception_model=goal_perception_model_M,
                                 name="M_block")

    # meta block
    return ChainBlockAgency(
        blocks = [D_block_agency,M_block_agency],
        start_block_id=0,
        end_block_id=1,
        transitions=np.array([[0,1],
                              [0,0]]),
        blocks_attention=[1,1]
    )


def yoked_anticipate(agent:ChainBlockAgency, obs:np.ndarray, observed_action:np.ndarray, **find_action_kwargs):
    """
    Own decision loop (not executed), then expectation of the observed action's outcome.

    Returns
    -------
    own_target : np.ndarray
        Target its own D block intended.
    own_action : np.ndarray
        Command its own M block selected (not executed).
    expected_target : np.ndarray
        Final position expected from the observed command.
    """
    D_block, M_block = agent.blocks

    # 1. own decision loop: sets goals, convergence and predictions of its own choice
    own_action = agent.find_action(obs,**find_action_kwargs)
    own_target = agent.stored_actions[0]

    # 2. expectation of the outcome of the other's command (replaces own-choice predictions)
    expected_target = M_block.forward_model(observed_action[np.newaxis,:])[0]
    M_block.predictions[-1] = expected_target
    D_block.predictions[-1] = D_block.forward_model(expected_target[np.newaxis,:])[0]
    return own_target, own_action, expected_target


def yoked_update(agent:ChainBlockAgency, observed_action:np.ndarray, obs:np.ndarray):
    """Learn from the observed command and outcome, attributing actions to blocks (see header)."""
    realized_target = _M_translate_states_to_goal(obs)
    agent.stored_actions = {0:realized_target, 1:observed_action}
    agent.update(observed_action,obs)


def history_to_dataframe(history:dict)->pd.DataFrame:
    raw_data = pd.DataFrame(np.stack(history["pos_state"]))
    raw_data = raw_data.rename(columns={j:f"pos_state_{j}" for j in range(raw_data.shape[1])})

    _raw_data = pd.DataFrame(np.stack(history["action"]))
    _raw_data = _raw_data.rename(columns={j:f"action_{j}" for j in range(_raw_data.shape[1])})
    raw_data = pd.concat([raw_data,_raw_data],axis=1)

    raw_data["episode"] = history["episode"]
    raw_data["reward"] = history["reward"]
    raw_data["trial"] = np.arange(len(raw_data))

    _raw_data = pd.DataFrame([ucommon.flatten(soa,keep_parent_key=False) for soa in history["sense_of_agency"]])
    raw_data = pd.concat([raw_data,_raw_data],axis=1)
    return raw_data


# agents
master_agent = build_agent() # acts
yoked_agent = build_agent()  # passively feels the master's actions and observations

# fit

obs, _ = env.reset() # LinearDampingManyTargetsEnv.reset returns (obs, reward)
master_history = {"episode":[],"pos_state":[],"action":[],"reward":[], "sense_of_agency":[]}
yoked_history = {"episode":[],"pos_state":[],"action":[],"reward":[], "sense_of_agency":[], "own_target":[], "own_action":[], "expected_target":[]}
n_episodes = 20
n_steps = 3
n_display = 1
for episode in range(n_episodes):
    for step in range(n_steps):
        pos_state = _M_translate_states_to_goal(obs)

        # master acts
        action = master_agent.find_action(obs,n_iters=10,n_bootstrap=10,n_mutations=10)

        # yoked forms its own decision (not executed) and expects the outcome of the other's action
        own_target, own_action, expected_target = yoked_anticipate(yoked_agent,obs,action,n_iters=10,n_bootstrap=10,n_mutations=10)

        obs,reward = env.step(action)

        master_agent.update(action,obs)
        yoked_update(yoked_agent,action,obs)

        for history,agent in [(master_history,master_agent),(yoked_history,yoked_agent)]:
            history["episode"].append(episode)
            history["pos_state"].append(pos_state)
            history["action"].append(action)
            history["reward"].append(reward)
            history["sense_of_agency"].append(agent.report_sense_of_agency())
        yoked_history["own_target"].append(own_target)
        yoked_history["own_action"].append(own_action)
        yoked_history["expected_target"].append(expected_target)

        if (episode % n_display) == 0:
            print(f"episode {episode+1}/{n_episodes} - {step+1}/{n_steps}: final pos = {obs[-2:]} -- action = {action} -- reward = {reward}")
            print(f"yoked: own target = {own_target} -- own action = {own_action} -- expected final pos = {expected_target}")
            print("master:",master_history["sense_of_agency"][-1])
            print("yoked: ",yoked_history["sense_of_agency"][-1])
            print()
    obs, _ = env.reset()

##################
folder_name = "LinearDampeningManyTargetsEnv_SeveralActions"
model_name = "model_GP2_yoked"

# dataframes
master_data = history_to_dataframe(master_history)
yoked_data = history_to_dataframe(yoked_history)
for key in ["own_target","own_action","expected_target"]:
    _values = np.stack(yoked_history[key])
    for j in range(_values.shape[1]):
        yoked_data[f"{key}_{j}"] = _values[:,j]

master_data.to_csv(f"data/{folder_name}/{model_name}_master.csv",index=False)
yoked_data.to_csv(f"data/{folder_name}/{model_name}_yoked.csv",index=False)

# plots
# master vs yoked on the same trials
udisplay.plot_values_across_trials([master_data,yoked_data],groups=["master (active)","yoked (passive)"],
                                   y_names=["reward","convergence_selected_action","goal_closure","competence_progress","prediction_error","learning_progress","feeling_of_effort"],
                                   sequence_groups="episode",sequence_style="background",sequence_labels=True,
                                   file_path=f"data/{folder_name}/{model_name}.html")
udisplay.plot_positions_to_actions_2D(yoked_history["pos_state"],yoked_history["action"], yoked_history["reward"],groups=yoked_history["episode"],colorbar_title="reward",file_path=f"data/{folder_name}/{model_name}_pos_actions.html")
