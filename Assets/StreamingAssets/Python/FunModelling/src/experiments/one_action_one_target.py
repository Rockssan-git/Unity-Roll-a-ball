# common librairies
import numpy as np
import pandas as pd

# my scripts
from src.env.toy_envs import LinearDampingSimpleTargetEnv
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
action_bounds = np.array([[-10.,10.],
                          [-10.,10.]])

# env
env = LinearDampingSimpleTargetEnv(
    action_bounds
)
obs = env.reset()


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
                           action_bounds=action_bounds,
                           forward_model=forward_model,
                           goal_perception_model=goal_perception_model_general,
                           name="M_block")


####################################################################################################################################
# play
####################################################################################################################################
history = {"action":[],"reward":[], "sense_of_agency":[]}
n_episodes = 20
n_display = 1
for episode in range(n_episodes): 
    action = M_block_agency.find_action(obs,n_iters=10,n_bootstrap=10,n_mutations=10)
    obs,reward = env.step(action)
    M_block_agency.update(action,obs)

    
    history["action"].append(action)
    history["reward"].append(reward)
    history["sense_of_agency"].append(M_block_agency.report_sense_of_agency())

    if (episode % n_display) == 0:
        print(f"episode {episode+1}/{n_episodes}: obs = {obs} -- action = {action} -- reward = {reward}")
        print(history["sense_of_agency"][-1])
        print()

####################################################################################################################################
# display
####################################################################################################################################

folder_name = "LinearDampeningSimpleTargetEnv"
model_name = "M_block_GP2"

udisplay.plot_2D_actions(history["action"], history["reward"],colorbar_title="reward",file_path=f"data/{folder_name}/{model_name}_actions.html")

raw_data = pd.DataFrame([[history["action"][i][j] for j in range(len(history["action"][i]))] for i in range(len(history["action"]))])
raw_data["reward"] = history["reward"]
raw_data["trial"] = np.arange(len(raw_data))
raw_data = raw_data.rename(columns={j:f"action_{j}" for j in range(len(history["action"][0]))})

_raw_data = pd.DataFrame([ucommon.flatten(history["sense_of_agency"][i],keep_parent_key=False) for i in range(len(history["action"]))])
raw_data = pd.concat([raw_data,_raw_data],axis=1)

udisplay.plot_values_across_trials(raw_data,y_names=["reward","convergence_selected_action","goal_closure","competence_progress","prediction_error","learning_progress","feeling_of_effort"],file_path=f"data/{folder_name}/{model_name}_agency.html")