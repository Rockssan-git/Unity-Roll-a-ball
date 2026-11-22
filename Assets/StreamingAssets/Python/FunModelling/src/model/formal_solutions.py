"""
Hand-coded agents (no learning) for the ArrowRoll Unity build.

Observations are assumed to start with the 3D target position ``obs[:3]``
followed by the 3D agent position ``obs[3:6]``.
"""
# common librairies
import numpy as np

# my scripts
from src.model.base import Agent


class OneDirectionAgent(Agent):
    """
    Pushes straight toward the target: action = target - position, clipped.

    Parameters
    ----------
    action_ids : list[int], default [0, 2]
        Axes of the 3D position that are controlled (x and z).
    action_bounds : list[float], default [-1, 1]
        ``[low, high]`` applied to every action dimension.
    """
    def __init__(self,action_ids: list[int] = [0,2], action_bounds:list[float] = [-1,1]):
        self.action_ids = action_ids
        self.action_bounds = action_bounds

    def find_action(self,obs:np.ndarray|list[np.ndarray]) -> np.ndarray:
        """``clip(target - position)`` on the controlled axes."""
        target,pos,_ = obs[:3],obs[3:6],obs[6:]
        action = (target - pos)[self.action_ids]
        action = np.clip(action,a_min=self.action_bounds[0],a_max=self.action_bounds[1])
        return action

    def forward_model(self,obs):
        pass


class LinearDampingAgent(OneDirectionAgent):
    """
    Exact command under linear damping: a constant command a moves the agent by
    ``a / d**2``, so ``a = d**2 * (target - position)``, clipped.

    Parameters
    ----------
    linear_damping : float
        Damping coefficient d of the environment.
    action_ids, action_bounds
        As in :class:`OneDirectionAgent`.
    """
    def __init__(self, linear_damping:float, action_ids: list[int] = [0,2], action_bounds:list[float] = [-1,1]):
        super().__init__(action_ids,action_bounds)
        self.linear_damping = linear_damping

    def find_action(self,obs:np.ndarray|list[np.ndarray]) -> np.ndarray:
        """``clip(d**2 * (target - position))`` on the controlled axes."""
        target,pos,_ = obs[:3],obs[3:6],obs[6:]
        action = self.linear_damping**2*(target - pos)[self.action_ids]
        action = np.clip(action,a_min=self.action_bounds[0],a_max=self.action_bounds[1])
        return action