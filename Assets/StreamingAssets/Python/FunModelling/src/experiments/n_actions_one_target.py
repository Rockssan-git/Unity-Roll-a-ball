# common librairies
import numpy as np
import pandas as pd


#from mlagents_envs.environment import UnityEnvironment
#from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
#from gym import Env

# my scripts
import src.utils.common as ucommon
import src.utils.display as udisplay

from src.env.toy_envs import LinearDampingSimpleTargetEnv
from src.model.block_agency import BlockAgency, ChainBlockAgency
from src.model.base import MaxGoalPerceptionModel, GoalPerceptionModel
from src.model.forward_model import GaussianProcessForwardModel


################################################################################
# Env = n actions, 1 target
# Model block D = find 1 target position, block M = get to position
################################################################################
action_bounds = np.array([[-2.,2.],
                          [-2.,2.]])

# Env
env = LinearDampingSimpleTargetEnv(
    action_bounds=action_bounds,
    several_steps=True
)

# D block = decides next position

forward_model = GaussianProcessForwardModel(out_dim=1) # predict score

goal_perception_model_general = MaxGoalPerceptionModel()

# (assumption: internalized score system)
def _D_translate_states_to_goal(states:np.ndarray):
    _states = states[...,2*1:] # 1 goal_position
    n_hits = env.compute_reward(overide_states=_states)
    return n_hits

goal_perception_model_general.translate_states_to_goal = _D_translate_states_to_goal
goal_perception_model_general.translate_predictions_to_goal = lambda x:x
goal_perception_model_general.translate_states_to_predictions = _D_translate_states_to_goal

D_block_agency = BlockAgency(goal=-np.array([np.nan]), # not used with MaxGoalPerceptionModel
                           action_bounds=action_bounds, # careful here: positions
                           forward_model=forward_model,
                           goal_perception_model=goal_perception_model_general,
                           name="D_block")


# M block = decides 2D command from a 2D position

forward_model = GaussianProcessForwardModel(out_dim=2) # predict position

goal_perception_model_general = GoalPerceptionModel()

# (assumption: internalized score system)
def _M_translate_states_to_goal(states:np.ndarray):
    _states = states[...,2*1:] # 1 goal_position so +2, states = only 2D final position 
    return _states

goal_perception_model_general.translate_states_to_goal = _M_translate_states_to_goal
goal_perception_model_general.translate_predictions_to_goal = lambda x:x
goal_perception_model_general.translate_states_to_predictions = _M_translate_states_to_goal

M_block_agency = BlockAgency(goal=np.array([1,1]), # changed online from D block
                           action_bounds=action_bounds, # careful here: take same bounds as D block but not same nature (D block's action = position, M block's action = command)
                           forward_model=forward_model,
                           goal_perception_model=goal_perception_model_general,
                           name="M_block")

# meta block
model_agent = ChainBlockAgency(
    blocks = [D_block_agency,M_block_agency],
    start_block_id=0,
    end_block_id=1,
    transitions=np.array([[0,1],
                          [0,0]]),
    blocks_attention=[1,1]
)

# fit

obs = env.reset()
history = {"episode":[],"pos_state":[],"action":[],"reward":[], "sense_of_agency":[]}
n_episodes = 20
n_steps = 2
n_display = 1
for episode in range(n_episodes):
    for step in range(n_steps):
        history["pos_state"].append(_M_translate_states_to_goal(obs))
        action = model_agent.find_action(obs,n_iters=10,n_bootstrap=10,n_mutations=10)
        obs,reward = env.step(action)
        model_agent.update(action,obs)
        
        history["episode"].append(episode)
        history["action"].append(action)
        history["reward"].append(reward)
        history["sense_of_agency"].append(model_agent.report_sense_of_agency())

        if (episode % n_display) == 0:
            print(f"episode {episode+1}/{n_episodes} - {step+1}/{n_steps}: obs = {obs} -- action = {action} -- reward = {reward}")
            print(history["sense_of_agency"][-1])
            print()
    obs = env.reset()

##################
folder_name = "LinearDampeningSimpleTargetsEnv_SeveralActions"
model_name = "model_GP2"

# dataframe
raw_data = pd.DataFrame([[history["pos_state"][i][j] for j in range(len(history["pos_state"][i]))] for i in range(len(history["pos_state"]))])
raw_data = raw_data.rename(columns={j:f"pos_state_{j}" for j in range(len(history["pos_state"][0]))})

_raw_data = pd.DataFrame([[history["action"][i][j] for j in range(len(history["action"][i]))] for i in range(len(history["action"]))])
_raw_data = _raw_data.rename(columns={j:f"action_{j}" for j in range(len(history["action"][0]))})
raw_data = pd.concat([raw_data,_raw_data],axis=1)

raw_data["episode"] = history["episode"]
raw_data["reward"] = history["reward"]
raw_data["trial"] = np.arange(len(raw_data))

_raw_data = pd.DataFrame([ucommon.flatten(history["sense_of_agency"][i],keep_parent_key=False) for i in range(len(history["action"]))])
raw_data = pd.concat([raw_data,_raw_data],axis=1)

# plots
udisplay.plot_values_across_trials(raw_data,y_names=["reward","convergence_selected_action","goal_closure","competence_progress","prediction_error","learning_progress","feeling_of_effort"],sequence_groups="episode",sequence_style="background",sequence_labels=True,file_path=f"data/{folder_name}/{model_name}.html")
udisplay.plot_2D_actions(history["action"], history["reward"],colorbar_title="reward",file_path=f"data/{folder_name}/{model_name}_actions.html")
udisplay.plot_positions_to_actions_2D(history["pos_state"],history["action"], history["reward"],groups=history["episode"],colorbar_title="reward",file_path=f"data/{folder_name}/{model_name}_pos_actions.html")