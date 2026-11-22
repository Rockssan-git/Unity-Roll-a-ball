# common librairies
import numpy as np
import pandas as pd

# my scripts
from src.env.toy_envs import LinearDampingManyTargetsEnv
from src.model.forward_model import GaussianProcessForwardModel
from src.model.base import GoalPerceptionModel, MaxGoalPerceptionModel
from src.model.block_agency import BlockAgency, ChainBlockAgency

import src.utils.display as udisplay
import src.utils.common as ucommon

####################################################################################################################################
# model definition
####################################################################################################################################

###########
# model 1 # M-module + D-module = predict only end point (simplification linear trajectory)
###########

# env
env = LinearDampingManyTargetsEnv(action_bounds=np.array([[-10.,10.],
                                                         [-10.,10.]])) # same bounds as the blocks below
obs, _ = env.reset() # LinearDampingManyTargetsEnv.reset returns (obs, reward)


# module M
forward_model = GaussianProcessForwardModel(out_dim=2) # predict final state
goal_perception_model_general = GoalPerceptionModel()
def _translate_to_goal(states):
    return states[...,-2:] # only final state taken

# (identical)
goal_perception_model_general.translate_states_to_goal = _translate_to_goal
goal_perception_model_general.translate_predictions_to_goal = lambda x:x
goal_perception_model_general.translate_states_to_predictions = _translate_to_goal


M_block_agency = BlockAgency(goal=np.array([1.,1.]), # will be adapted online
                           action_bounds=np.array([[-10.,10.],
                                                   [-10.,10.]]),
                           forward_model=forward_model,
                           goal_perception_model=goal_perception_model_general,
                           name="M_block")

# module D
forward_model = GaussianProcessForwardModel(out_dim=1) # only predict score
goal_perception_model_general = MaxGoalPerceptionModel()

# (assumption: internalized score system)
def _translate_states_to_goal(states):
    _states = states[...,2*len(env.goal_positions):]
    if len(_states.shape) < 2:
        _states = _states.reshape((-1,2))
    else:
        _states = _states.reshape((_states.shape[0],-1,2))
    n_hits = env.compute_reward(overide_states=_states)
    return n_hits


goal_perception_model_general.translate_states_to_goal = _translate_states_to_goal
goal_perception_model_general.translate_predictions_to_goal = lambda x:x
goal_perception_model_general.translate_states_to_predictions = _translate_states_to_goal

D_block_agency = BlockAgency(goal=-np.array([np.nan]), # not used with MaxGoalPerceptionModel
                           action_bounds=np.array([[-10.,10.],
                                                   [-10.,10.]]),
                           forward_model=forward_model,
                           goal_perception_model=goal_perception_model_general,
                           name="D_block")

# Aggregate blocks:
model_1_agent = ChainBlockAgency(
    blocks = [D_block_agency,M_block_agency],
    start_block_id=0,
    end_block_id=1,
    transitions=np.array([[0,1],
                          [0,0]]),
    blocks_attention=[1,1]
)

# TODO: do version 2 with trajectory representation (create a class for this and internal comparison)

####################################################################################################################################
# play
####################################################################################################################################
history = {"action":[],"reward":[], "sense_of_agency":[]}
n_episodes = 20
n_display = 1
for episode in range(n_episodes): 
    action = model_1_agent.find_action(obs,n_iters=10,n_bootstrap=10,n_mutations=10)
    obs,reward = env.step(action)
    model_1_agent.update(action,obs)
    history["action"].append(action)
    history["reward"].append(reward)
    history["sense_of_agency"].append(model_1_agent.report_sense_of_agency())

    if (episode % n_display) == 0:
        print(f"episode {episode+1}/{n_episodes}: obs = {obs} -- action = {action} -- reward = {reward}")
        print(history["sense_of_agency"][-1])
        print()

####################################################################################################################################
# display
####################################################################################################################################

folder_name = "LinearDampeningManyTargetsEnv"
model_name = "model1_GP2"

udisplay.plot_2D_actions(history["action"], history["reward"],colorbar_title="reward",file_path=f"data/{folder_name}/{model_name}_actions.html")

raw_data = pd.DataFrame([[history["action"][i][j] for j in range(len(history["action"][i]))] for i in range(len(history["action"]))])
raw_data["reward"] = history["reward"]
raw_data["trial"] = np.arange(len(raw_data))
raw_data = raw_data.rename(columns={j:f"action_{j}" for j in range(len(history["action"][0]))})

_raw_data = pd.DataFrame([ucommon.flatten(history["sense_of_agency"][i],keep_parent_key=False) for i in range(len(history["action"]))])
raw_data = pd.concat([raw_data,_raw_data],axis=1)

udisplay.plot_values_across_trials(raw_data,y_names=["reward","convergence_selected_action","goal_closure","competence_progress","prediction_error","learning_progress","feeling_of_effort"],file_path=f"data/{folder_name}/{model_name}_agency.html")