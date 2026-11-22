"""
Toy 2D environments: a point pushed by a constant command against a linear damping.

They are Python stand-ins for the ArrowRoll Unity game.
"""
# common librairies
import numpy as np
from scipy.spatial.distance import cdist


class LinearDampingSimpleTargetEnv:
    """
    One target at (1, 1); a step moves the point by ``action / linear_damping**2``.

    Parameters
    ----------
    action_bounds : np.ndarray, shape (2, 2)
        ``[low, high]`` for each action dimension; actions are clipped to it.
    linear_damping : float, default 0.45
        Damping coefficient d.
    several_steps : bool, default False
        If True, each step starts where the previous one ended; otherwise every
        step starts from the origin.
    """
    def __init__(self,action_bounds:np.ndarray,linear_damping:float=.45,several_steps:bool=False):
        self.action_bounds = action_bounds
        self.linear_damping = linear_damping
        self.several_steps = several_steps

        self.state_init = np.zeros(2)
        self.last_state = self.state_init
        self.goal_state = np.array([1,1])

        self.state = None


    def compute_reward(self,overide_states:np.ndarray|None=None)->float:
        """
        Reward ``1 / (1 + ||state - goal||)``.

        Parameters
        ----------
        overide_states : np.ndarray, shape (2,) or (n_states, 2), optional
            States to score instead of the current one.

        Returns
        -------
        float or np.ndarray
            One reward, or one per state when several are given.
        """
        if overide_states is not None:
            _states = overide_states
        else:
            _states = self.state

        if _states is None:
            raise AttributeError("Cannot call compute_reward before step")

        if len(_states.shape) == 1:
            _states = _states.reshape((1,_states.shape[0])) # (n_states,dim)

        dist = np.linalg.norm(_states-self.goal_state,axis=-1) # (n_states,dim) - (dim,) -> (n_states,dim) 

        if len(dist) == 1:
            dist = dist[0]

        return 1/(dist+1)


    def step(self,action:np.ndarray):
        """
        Apply one command.

        Parameters
        ----------
        action : np.ndarray, shape (2,)
            Command, clipped to ``action_bounds``.

        Returns
        -------
        obs : np.ndarray, shape (4,)
            Goal position followed by the new position.
        reward : float
        """
        assert action.shape == (2,)
        _action = np.clip(action,self.action_bounds[:,0],self.action_bounds[:,1])

        self.state = self.last_state + _action/(self.linear_damping**2)

        if self.several_steps:
            self.last_state = self.state

        reward = self.compute_reward()
        return np.concatenate([self.goal_state,self.state]),reward


    def reset(self):
        """
        Put the point back at the origin.

        Returns
        -------
        obs : np.ndarray, shape (4,)
            Goal position followed by the initial position (no reward).
        """
        self.state = None
        self.last_state = self.state_init
        return np.concatenate([self.goal_state,self.state_init])



class LinearDampingManyTargetsEnv:
    """
    Five targets; a step simulates ``x'' = a - d x'`` for ``max_step_time`` seconds at 24 Hz.

    The reward is the number of targets the trajectory passes within 0.1 of.

    Parameters
    ----------
    action_bounds : np.ndarray, shape (2, 2)
        ``[low, high]`` for each action dimension; actions are clipped to it.
    linear_damping : float, default 0.45
        Damping coefficient d.
    several_steps : bool, default False
        If True, each step starts where the previous one ended; otherwise every
        step starts from the origin.
    """
    def __init__(self,action_bounds:np.ndarray,linear_damping:float=.45,several_steps:bool=False):
        self.action_bounds = action_bounds
        self.linear_damping = linear_damping
        self.several_steps = several_steps

        self.state_init = np.zeros(2) # should be later of shape (n_steps/episode,2)
        self.last_pos_state = self.state_init
        self.speed_init = np.zeros(2)
        self.goal_positions = np.array([[1,1],
                                     [2,2],
                                     [1,0],
                                     [2,0],
                                     [3,3]])
        
        self.max_step_time = 10 # in seconds

        self.states = None


    def compute_reward(self,overide_states:np.ndarray|None=None)->float:
        """
        Number of targets hit: targets that some point of the trajectory comes within 0.1 of.

        Parameters
        ----------
        overide_states : np.ndarray, shape (n_t, 2) or (n_sequences, n_t, 2), optional
            Trajectories to score instead of the current one.

        Returns
        -------
        int or np.ndarray
            Number of hits, or one per trajectory when several are given.
        """
        if overide_states is not None:
            _states = overide_states
        else:
            _states = self.states

        assert not(_states is None), "Cannot call compute_reward before step"

        if len(_states.shape) == 2:
            _states = _states.reshape((1,_states.shape[0],_states.shape[1])) # (n_episodes, n_steps/episode,dim)

        assert len(_states.shape) == 3, f"_states.shape: {_states.shape}"

        n_sequences,n_steps,dim = _states.shape
        _states = _states.reshape((n_sequences*n_steps,dim))
        dist = cdist(_states,self.goal_positions) # each (sequence,step) compared to goals
        dist = dist.reshape((n_sequences,n_steps,len(self.goal_positions))) # return initial shape
        dist = np.min(dist,axis=1) # (n_sequences, n_goals)
        dist = np.sum(dist<.1,axis=1) # ~ count hits, shape = n_sequences

        if len(dist) == 1:
            dist = dist[0]
        
        return dist


    def step(self,action:np.ndarray):
        """
        Apply one command for ``max_step_time`` seconds.

        Parameters
        ----------
        action : np.ndarray, shape (2,)
            Command, clipped to ``action_bounds``.

        Returns
        -------
        obs : np.ndarray, shape (2 * (n_goals + n_t),)
            Flattened goal positions ``(5, 2)`` followed by the trajectory ``(241, 2)``.
        reward : int
            Number of targets hit.
        """
        assert action.shape == (2,)
        _action = np.clip(action,self.action_bounds[:,0],self.action_bounds[:,1])

        t_range = np.linspace(0,self.max_step_time,num=self.max_step_time*24+1) # 24Hz resolution

        term0 = 1/self.linear_damping * t_range[:,np.newaxis]@_action[np.newaxis,:]

        fac0 = (1-np.exp(-self.linear_damping*t_range))[:,np.newaxis]
        # (a/d^2) * (1 - (d/a) * v0) rewritten as a/d^2 - v0/d to avoid dividing by the action
        fac1 = (_action/(self.linear_damping**2) - self.speed_init/self.linear_damping)[np.newaxis,:]
        term1 = fac0@fac1

        self.states = term0 - term1 + self.last_pos_state

        if self.several_steps:
            self.last_pos_state = self.states[-1]

        reward = self.compute_reward()

        return np.concatenate([self.goal_positions,self.states],axis=0).flatten(),reward


    def reset(self):
        """
        Put the point back at the origin.

        Returns
        -------
        obs : np.ndarray
            Same layout as :meth:`step`, with the trajectory staying at the origin.
        reward : int
            Number of targets hit by that still trajectory.
        """
        self.last_pos_state = self.state_init
        n_t = self.max_step_time*24+1 # same time resolution as step
        init_states = np.tile(self.state_init,(n_t,1))
        reward = self.compute_reward(overide_states=init_states)
        self.states = None
        return np.concatenate([self.goal_positions,init_states],axis=0).flatten(),reward