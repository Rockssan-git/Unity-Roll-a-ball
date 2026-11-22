# common librairies
import numpy as np

# my scripts
from src.model.base import Agent, GoalPerceptionModel
from src.model.forward_model import ForwardModel

#################################################################################################################################################################################################
#################################################################################################################################################################################################
# Basic block
#################################################################################################################################################################################################
#################################################################################################################################################################################################

class BlockAgency(Agent):
    """
    Basic learning unit: learns a forward model, picks the action whose predicted
    outcome is closest to its goal, and tracks sense-of-agency signals.

    Parameters
    ----------
    goal : np.ndarray
        Target in goal space. Can be changed between calls (e.g. by a
        :class:`ChainBlockAgency`); ignored with a :class:`~src.model.base.MaxGoalPerceptionModel`.
    action_bounds : np.ndarray, shape (action_dim, 2)
        ``[low, high]`` for each action dimension.
    forward_model : ForwardModel
        Predicts the outcome of an action.
    goal_perception_model : GoalPerceptionModel
        Maps observations/predictions to goal space and measures the distance to the goal.
    window_size : int, default 4
        Number of recent trials used for competence and learning progress.
    name : str, default ""
        Label, for debugging.
    """
    def __init__(self, goal:np.ndarray, action_bounds:np.ndarray, forward_model:ForwardModel, goal_perception_model:GoalPerceptionModel, window_size:int=4, name:str=""):
        # signals and modules
        self.goal = goal
        self.action_bounds = action_bounds
        self.action_dim = action_bounds.shape[0]
        self.forward_model = forward_model
        self.goal_perception_model = goal_perception_model
        self.window_size = window_size
        self.name = name # useful for debugging and interpreting

        # memory
        self.prev_action = None
        self.states = []
        self.predictions = []
        self.errors = []

        # sense of agency components
        ## sense of initiation
        self.convergence_selected_action = None

        ## sense of control
        self.goal_closure = None
        self.competence_progress = None
        self.prediction_error = None
        self.learning_progress = None
        self.feeling_of_effort = None        

        ## sense of intentional causation
        # nothing else

        self.update_sense_of_agency()

    def update_sense_of_agency(self):
        """Rebuild the ``sense_of_agency`` dict from the current signal values."""
        self.sense_of_initiation = {"convergence_selected_action":self.convergence_selected_action}

        self.sense_of_control = {
            "goal_closure":self.goal_closure,
            "competence_progress":self.competence_progress,
            "prediction_error":self.prediction_error,
            "learning_progress":self.learning_progress,
            "feeling_of_effort":self.feeling_of_effort
        }
        
        ## sense of intentional causation
        self.sense_of_intentional_causation = {
            "goal_closure":self.goal_closure
        }
        
        ## aggregation
        self.sense_of_agency = {
            "sense_of_initiation": self.sense_of_initiation,
            "sense_of_control": self.sense_of_control,
            "sense_of_intentional_causation":self.sense_of_intentional_causation
        }

    def compute_competence_progress(self):
        """
        Change in goal closure over the last ``window_size`` trials (naive/window implementation):
        mean of the recent half minus mean of the older half. 0 until ``window_size`` trials are stored.
        """
        if len(self.states) < self.window_size:
            return 0
        else:
            goal_closures = self.goal_perception_model.distance_states_to_goal(np.stack(self.states),self.goal)

            restricted_goal_closures = goal_closures[-self.window_size:]
            
            w = self.window_size//2+self.window_size%2
            diff = np.mean(restricted_goal_closures[-w:]) - np.mean(restricted_goal_closures[:w])
            return diff
    

    def compute_learning_progress(self):
        """
        Change in prediction error over the last ``window_size`` trials (naive/window implementation):
        mean of the recent half minus mean of the older half. 0 until ``window_size`` trials are stored.
        """
        if len(self.errors) < self.window_size:
            return 0
        else:
            restricted_errors = self.errors[-self.window_size:]

            w = self.window_size//2+self.window_size%2
            diff = np.mean(restricted_errors[-w:]) - np.mean(restricted_errors[:w])
            return diff


    def update(self,action:np.ndarray,obs:np.ndarray):
        """
        Learn from the outcome of ``action``: update the agency signals, then refit the forward model.

        Must be called after :meth:`find_action`, whose prediction is used for the prediction error.

        Parameters
        ----------
        action : np.ndarray
            Action this block chose.
        obs : np.ndarray
            Observation that followed.
        """
        self.goal_closure =self.goal_perception_model.distance_states_to_goal(obs,self.goal)
        self.errors.append(np.linalg.norm(self.goal_perception_model.translate_states_to_predictions(obs) - self.predictions[-1])) # should be for forward_model to compute?
        self.prediction_error = self.errors[-1]

        self.states.append(obs) # WARNING: could have obs <-> state conversion required 
        self.competence_progress = self.compute_competence_progress()
        self.learning_progress = self.compute_learning_progress()

        if self.prev_action is None:
            self.feeling_of_effort = 0
        else:
            self.feeling_of_effort = np.linalg.norm(action - self.prev_action)
        
        self.prev_action = action

        self.update_sense_of_agency()
        
        self.forward_model.update(action,self.goal_perception_model.translate_states_to_predictions(obs))


    def objective_func(self,predictions:np.ndarray):
        """Score to minimise: distance of each prediction ``(n, out_dim)`` to the goal."""
        reshaped_goal = np.ones((predictions.shape[0],1))@self.goal.reshape((1,self.goal.size))
        return self.goal_perception_model.distance_predictions_to_goal(predictions,reshaped_goal) # euclidian distance


    def random_sample(self,n_samples:int)->np.ndarray:
        """Draw ``n_samples`` actions uniformly within ``action_bounds`` -> ``(n_samples, action_dim)``."""
        return np.random.sample((n_samples,self.action_dim))*(self.action_bounds[:,1]-self.action_bounds[:,0])+self.action_bounds[:,0]


    def mutate(self,actions:np.ndarray,n_mutations:int,ratio_std:float=1/20):
        """
        Make ``n_mutations`` noisy copies of each action.

        The noise is a uniform sample within ``action_bounds`` scaled by ``ratio_std``.
        Mutated actions are not clipped to the bounds.

        Returns
        -------
        np.ndarray, shape (n_mutations * len(actions), action_dim)
        """
        new_actions = []
        for _ in range(n_mutations):
            added_mutation = ratio_std*self.random_sample(actions.shape[0])
            new_actions.append(actions + added_mutation)

        new_actions = np.concatenate(new_actions,axis=0)
        return new_actions


    def find_action(self,obs:np.ndarray|list[np.ndarray],n_iters:int,n_bootstrap:int,n_mutations:int,top_ratio_kept:float=.1,min_delta:float=1e-3):
        """
        Evolutionary search for the action minimising :meth:`objective_func` on the forward model's predictions.

        Also stores the prediction of the chosen action (used by :meth:`update`) and sets
        ``convergence_selected_action``.

        Parameters
        ----------
        obs : np.ndarray
            Current observation (not used by the search itself).
        n_iters : int
            Maximum number of keep-and-mutate iterations.
        n_bootstrap : int
            Number of random initial candidates.
        n_mutations : int
            Noisy copies made of each kept candidate per iteration.
        top_ratio_kept : float, default 0.1
            Fraction of the candidates kept at each iteration.
        min_delta : float, default 1e-3
            Stop when the best score changes by less than this.

        Returns
        -------
        np.ndarray, shape (action_dim,)
            Best action found.
        """
        # create random samples in the action space
        candidates = self.random_sample(n_bootstrap)
        candidates_predictions = self.forward_model(candidates)
        # - evaluate output with forward model and distance to goal_signal
        scores = self.objective_func(candidates_predictions)
        
        # external loop
        best_candidate,best_score = None,None
        for i in range(n_iters):
            # - pick top_ratio_kept samples
            n_kept = int(len(scores)*top_ratio_kept)
            kept_idx = np.argsort(scores)[:n_kept]
            kept_candidates = candidates[kept_idx].reshape((-1,self.action_dim))
            kept_predictions = candidates_predictions[kept_idx].reshape((-1,self.forward_model.out_dim))
            kept_scores = scores[kept_idx].flatten() # always 1 dim for objective_func

            # - if convergence: break (here because avoids computing argmin)   
            ## convergence: less than min_delta improvement
            best_candidate = candidates[kept_idx[0]] 
            
            if not(best_score is None):
                self.convergence_selected_action = np.abs(best_score-scores[kept_idx[0]]).item()
                if self.convergence_selected_action < min_delta:
                    best_score = scores[kept_idx[0]]
                    break
            else:
                best_score = scores[kept_idx[0]]

            # - make variations of kept samples and evaluate
            new_candidates = self.mutate(kept_candidates,n_mutations)
            new_predictions = self.forward_model(new_candidates)
            new_scores = self.objective_func(new_predictions).flatten()

            # - merge candidates
            #print("kept_candidates\n",kept_candidates,kept_candidates.shape)
            #print("new_candidates\n",new_candidates,new_candidates.shape)
            candidates = np.concatenate([kept_candidates,new_candidates],axis=0)
            #print("kept_predictions\n",kept_predictions,kept_predictions.shape)
            #print("new_predictions\n",new_predictions,new_predictions.shape)
            candidates_predictions = np.concatenate([kept_predictions,new_predictions],axis=0)
            #print("kept_scores\n",kept_scores,kept_scores.shape)
            #print("new_scores\n",new_scores,new_scores.shape)
            scores = np.concatenate([kept_scores,new_scores],axis=0)

        self.predictions.append(candidates_predictions[kept_idx[0]])
        return best_candidate

    def report_sense_of_agency(self):
        """
        Current agency signals, as
        ``{"sense_of_initiation": {...}, "sense_of_control": {...}, "sense_of_intentional_causation": {...}}``.
        """
        return self.sense_of_agency

#################################################################################################################################################################################################
#################################################################################################################################################################################################
# Hierarchy
#################################################################################################################################################################################################
#################################################################################################################################################################################################

class ChainBlockAgency(Agent):
    """
    Chain of blocks: the action chosen by one block becomes the goal of the next,
    and the last block's action is sent to the environment.

    The chain is encoded as a general graph for later implementations of block hierarchies.

    Parameters
    ----------
    blocks : list[BlockAgency]
    start_block_id : int
        Index of the first block (its own ``goal`` is used as the starting goal).
    end_block_id : int
        Index of the last block, whose action is returned.
    transitions : np.ndarray, shape (n_blocks, n_blocks)
        Adjacency matrix: ``transitions[i, j] = 1`` if block i sends its action to block j.
        Must describe a chain ending at ``end_block_id``.
    blocks_attention : list[float], optional
        Weight of each block when summing agency signals. Defaults to 1 for every block.
    """
    def __init__(self, blocks:list[BlockAgency], start_block_id:int, end_block_id:int,transitions:np.ndarray, blocks_attention:list[float]|None=None):
        self.blocks = blocks
        self.start_block_id = start_block_id
        self.end_block_id = end_block_id

        assert self._check_transitions(transitions)
        self.transitions = transitions

        self.graph_dict = self._convert_graph_to_dict(transitions)
        self.stored_actions = {} # dict node_id:action or goal sent

        self.blocks_attention = blocks_attention if blocks_attention is not None else [1. for _ in range(len(blocks))] 
        assert len(self.blocks_attention) == len(blocks)


    def _check_transitions(self, transitions:np.ndarray):
        """True if ``transitions`` is a chain: every block but the end one has exactly one successor."""
        # NotImplemented and more of a utils function
        chain_dict = self._convert_graph_to_dict(transitions)

        # check end
        if len(chain_dict[self.end_block_id]) != 0:
            return False

        # check transition is a chain
        for i in chain_dict:
            if len(chain_dict[i]) != 1 and i != self.end_block_id:
                return False

        return True


    def _convert_graph_to_dict(self,transitions:np.ndarray):
        """Adjacency matrix -> ``{block_id: [successor ids]}``."""
        graph_dict = {i:[] for i in range(transitions.shape[0])}
        senders, receivers = np.where(transitions)
        for s,r in zip(senders,receivers):
            graph_dict[s].append(r)
        return graph_dict


    def _next_nodes(self,node_id:int)->list[int]:
        return self.graph_dict[node_id]


    def find_action(self, obs:np.ndarray,**kwargs):
        """
        Run the blocks in order, each one's action becoming the next one's goal.

        Assumes a tree structure, and passes the same ``kwargs`` to every block's
        :meth:`BlockAgency.find_action`.

        Returns
        -------
        np.ndarray
            Action of the last block.
        """
        self.stored_actions = {} # flush
        messages = [(self.blocks[self.start_block_id].goal,self.start_block_id)] # format (goal,node_id) = (the message,receiving)
        leaves = []
        
        while len(messages):
            goal, node_id = messages.pop(0) # FIFO
            
            # find block
            block = self.blocks[node_id]

            block.goal = goal
            out = block.find_action(obs,**kwargs)

            # go to next node
            for next_node_id in self.graph_dict[node_id]:
                messages.append((out,next_node_id))

            if len(self.graph_dict[node_id]) == 0:
                leaves.append(out)

            # store
            self.stored_actions[node_id] = out

        # return to single output (chain)
        assert len(leaves) == 1
        return leaves[0]


    def update(self,action:np.ndarray,obs:np.ndarray):
        """
        Update every block with the action it chose in the last :meth:`find_action`
        and the observation ``obs``, then aggregate the agency signals.
        ``action`` itself is not used.
        """
        for node_id,node_action in self.stored_actions.items():
            self.blocks[node_id].update(node_action,obs)

        self.update_sense_of_agency()

    def _weighted_linear_combination(self,x:np.ndarray|list[int],w:np.ndarray|list[int]):
        np.array(x)@np.array(w)
        pass

    def update_sense_of_agency(self):
        """Sum each agency signal across blocks, weighted by ``blocks_attention``."""
        self.sense_of_initiation = {"convergence_selected_action":np.array([b.convergence_selected_action for b in self.blocks])@np.array(self.blocks_attention)}
        
        self.sense_of_control = {
            "goal_closure":np.array([b.goal_closure for b in self.blocks])@np.array(self.blocks_attention),
            "competence_progress":np.array([b.competence_progress for b in self.blocks])@np.array(self.blocks_attention),
            "prediction_error":np.array([b.prediction_error for b in self.blocks])@np.array(self.blocks_attention),
            "learning_progress":np.array([b.learning_progress for b in self.blocks])@np.array(self.blocks_attention),
            "feeling_of_effort":np.array([b.feeling_of_effort for b in self.blocks])@np.array(self.blocks_attention)
        }
                
        ## sense of intentional causation
        self.sense_of_intentional_causation = {
            "goal_closure":np.array([b.goal_closure for b in self.blocks])@np.array(self.blocks_attention)
        }
                
        ## aggregation
        self.sense_of_agency = {
            "sense_of_initiation": self.sense_of_initiation,
            "sense_of_control": self.sense_of_control,
            "sense_of_intentional_causation":self.sense_of_intentional_causation
        }


    def report_sense_of_agency(self):
        """Aggregated agency signals, same layout as :meth:`BlockAgency.report_sense_of_agency`."""
        return self.sense_of_agency