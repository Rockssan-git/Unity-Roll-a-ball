"""Run an agent on the ArrowRoll Unity build through Unity ML-Agents and its Gym wrapper."""
# common libairies
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

# my scripts
from src.model.base import Agent

class Trainer:
    """
    Runs ``agent`` on a Unity executable.

    Parameters
    ----------
    agent : Agent
    env_path : str
        Path to the Unity executable (e.g. ``env/<build>/ArrowRoll_<date>.x86_64``).
    uint8_visual, flatten_branched, allow_multiple_obs
        Options of ``UnityToGymWrapper``.
    """
    # TODO: a helper for training agents
    def __init__(self, agent:Agent, env_path:str, uint8_visual=False, flatten_branched=True, allow_multiple_obs=False):
        self.env_path = env_path
        self.agent = agent

        self.uint8_visual=False, 
        self.flatten_branched=True, 
        self.allow_multiple_obs=False

    def run(self,n_steps:int,n_episodes:int=1,n_display:int = 1,seed:int|None=None):
        """
        Launch the Unity environment and run the agent (no learning update yet).

        Parameters
        ----------
        n_steps : int
            Maximum steps per episode.
        n_episodes : int, default 1
        n_display : int, default 1
            Print progress every ``n_display`` steps.
        seed : int, optional
            Seed passed to the Unity environment.
        """
        # load env
        unity_env = UnityEnvironment(file_name=self.env_path,seed=seed)
        env = UnityToGymWrapper(unity_env, uint8_visual=self.uint8_visual, flatten_branched=self.flatten_branched, allow_multiple_obs=self.allow_multiple_obs)
        
        # run
        for i_episode in range(n_episodes):
            print(f"\ni_episode: {i_episode+1}/{n_episodes}\n")
            obs, _ = env.reset()
            for i_step in range(n_steps):
                action = self.agent.find_action(obs=obs)
                obs, reward, terminated, _, _ = env.step(action)
                # TODO: miss update step here
                if terminated:
                    print(f"episode terminated -- i_step: {i_step+1}/{n_steps} -- reward:{reward}")
                    break

                if (i_step % n_display) == 0:
                    print(f"i_step: {i_step+1}/{n_steps} -- reward:{reward}")

                    # TODO:remove: assumes obs shape + distance = reward
                    target,pos = obs[:3],obs[3:6]
                    dist = np.linalg.norm(target-pos)
                    print(f"distance:{dist:.2f}")

        env.close()

if __name__ == "__main__":
    from src.model.formal_solutions import LinearDampingAgent,OneDirectionAgent

    # an example use
    agent = LinearDampingAgent(linear_damping=.45)
    agent = OneDirectionAgent()
    trainer = Trainer(agent=agent, env_path="env/2026-11-01_exe/ArrowRoll_2026-11-01.x86_64")
    trainer.run(100,n_display=5,seed=1)