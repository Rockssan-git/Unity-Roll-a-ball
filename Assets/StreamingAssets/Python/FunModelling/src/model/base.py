"""Base classes for agents and goal perception models."""
from abc import ABC,abstractmethod

import numpy as np

class Agent(ABC):
    """Anything that maps an observation to an action."""
    @abstractmethod
    def find_action(self,obs:np.ndarray|list[np.ndarray]) -> np.ndarray:
        """Return the action to take given observation ``obs``."""
        pass


class GoalPerceptionModel:
    """
    Maps observations and forward-model predictions into the space where a goal is
    defined, and measures their Euclidean distance to the goal.

    The ``translate_*`` methods are meant to be replaced, either by subclassing or by
    assigning functions to the attributes of an instance (as in the experiment scripts).
    """
    @abstractmethod
    def translate_predictions_to_goal(self,predictions:np.ndarray)->np.ndarray:
        """Forward-model predictions -> goal space."""
        pass

    @abstractmethod
    def translate_states_to_goal(self,states:np.ndarray)->np.ndarray:
        """Observed states -> goal space."""
        pass

    def translate_states_to_predictions(self,states:np.ndarray)->np.ndarray:
        """Observed states -> forward-model prediction space (identity by default, overide otherwise)."""
        return states

    def distance_states_to_goal(self,states:np.ndarray|list[np.ndarray], goal:np.ndarray) -> np.ndarray:
        """Euclidean distance between the observed states (in goal space) and ``goal``."""
        goal_states = self.translate_states_to_goal(states)
        return np.linalg.norm(goal_states - goal,axis=-1)

    def distance_predictions_to_goal(self,predictions:np.ndarray|list[np.ndarray], goal:np.ndarray) -> np.ndarray:
        """Euclidean distance between the predictions (in goal space) and ``goal``."""
        goal_preds = self.translate_predictions_to_goal(predictions)
        return np.linalg.norm(goal_preds - goal,axis=-1)

class MaxGoalPerceptionModel(GoalPerceptionModel):
    """
    Goal = maximise a score. The "distance" is minus the score in goal space, and
    ``goal`` is ignored.
    """
    # minus because, in this project, all optimization algorithms minimize
    def distance_states_to_goal(self,states:np.ndarray|list[np.ndarray], goal:np.ndarray) -> np.ndarray:
        """Minus the score of the observed states (``goal`` is ignored)."""
        goal_states = self.translate_states_to_goal(states)
        return -goal_states

    def distance_predictions_to_goal(self,predictions:np.ndarray|list[np.ndarray], goal:np.ndarray) -> np.ndarray:
        """Minus the predicted score (``goal`` is ignored)."""
        goal_preds = self.translate_predictions_to_goal(predictions)
        return -goal_preds
