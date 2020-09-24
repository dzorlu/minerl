
import json
import select
import time
import logging
import os
import threading


from typing import Callable

import aicrowd_helper
import gym
import minerl
import abc
import numpy as np
from pathlib import Path

from acme.agents.tf import actors
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme import wrappers
from acme import specs

import sonnet as snt
import tensorflow as tf

import coloredlogs, logging
coloredlogs.install(logging.WARNING)
logger = logging.getLogger(__name__)

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))

class EpisodeDone(Exception):
    pass

# class Episode(gym.Env):
#     """A class for a single episode.
#     """
#     def __init__(self, env):
#         self.env = env
#         self.action_space = env.action_space
#         self.observation_space = env.observation_space
#         self._done = False

#     def reset(self):
#         if not self._done:
#             return self.env.reset()

#     def step(self, action):
#         s,r,d,i = self.env.step(action)
#         if d:
#             self._done = True
#             raise EpisodeDone()
#         else:
#             return s,r,d,i

class Episode:
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        ts = self.env.step(action)
        if ts.last():
            self._done = True
            raise EpisodeDone()
        else:
            return ts



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.
    
    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class 

    This class enables the evaluator to run your agent in parallel, 
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs)) 
                ...
        
        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################
from acme.tf import networks
import dm_env
import functools
NUMBER_OF_DISCRETE_ACTIONS = 25 #make sure this matches train.py
rel_path = os.path.dirname(__file__) # relative directory path
model_dir = os.path.join(rel_path, "train")
Path(model_dir).mkdir(parents=True, exist_ok=True)
logger.info(model_dir)

def create_network(nb_actions: int = NUMBER_OF_DISCRETE_ACTIONS) -> networks.RNNCore:
    """Creates the policy network"""
    return networks.R2D2MineRLNetwork(nb_actions)


def make_environment(k_means_path: str,
                     num_actions: int = NUMBER_OF_DISCRETE_ACTIONS,
                     dat_loader: minerl.data.data_pipeline.DataPipeline = None, 
                     train: bool = True,
                     minerl_gym_env: str = MINERL_GYM_ENV) -> dm_env.Environment:
  """
  Wrap the environment in:
    1 - MineRLWrapper 
        - similar to OAR but add proprioceptive features
        - kMeans to map cont action space to a discrete one
    2 - SinglePrecisionWrapper
    3 - GymWrapper
  """

  env = gym.make(minerl_gym_env)
      
  return wrappers.wrap_all(env, [
      wrappers.GymWrapper,
      functools.partial(
        wrappers.MineRLWrapper,
          num_actions=num_actions,
          dat_loader=dat_loader,
          k_means_path=k_means_path,
          train=False
      ),
      wrappers.SinglePrecisionWrapper,
  ])

def load_actor(environment_spec):
    network = create_network(NUMBER_OF_DISCRETE_ACTIONS)
    tf2_utils.create_variables(network, [environment_spec.observations])
    # restores the model
    tf2_savers.Checkpointer(
        directory=model_dir, 
        subdirectory='r2d2_learner_v1',
        time_delta_minutes=15,
        objects_to_save={'network': network}, #only revive the network
    )

    policy_network = snt.DeepRNN([
        network,
        lambda qs: tf.math.argmax(qs, axis=-1), # this is different at inference time.
    ])
    actor = actors.RecurrentActor(policy_network, None)
    return actor



class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent. 
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64*64*3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size))*2 -1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten()/255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1,1)}


    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs,reward,done,_ = single_episode_env.step(self.act(self.flatten_obs(obs)))

class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env : Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)

class R2D3Agent(MineRLAgentBase):
    """A random agent"""
    def load_agent(self, actor):
        self.actor = actor

    def run_agent_on_episode(self, single_episode_env : Episode):
        ts = single_episode_env.reset()
        while not ts.last():
            action = self.actor.select_action(ts.observation)
            ts = single_episode_env.step(action)
        
#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   # 
######################################################################
AGENT_TO_TEST = R2D3Agent # MineRLMatrixAgent, MineRLRandomAgent, YourAgentHere



####################
# EVALUATION CODE  #
####################
def main():
    #
    environment = make_environment(num_actions=NUMBER_OF_DISCRETE_ACTIONS, k_means_path=model_dir)
    spec = specs.make_environment_spec(environment)
    actor = load_actor(spec) #initiate here to keep the state shared across threads
    environment.close()
    
    agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent(actor)

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    # Create the parallel envs (sequentially to prevent issues!)
    envs = list()
    for _ in range(EVALUATION_THREAD_COUNT):
        environment = make_environment(num_actions=NUMBER_OF_DISCRETE_ACTIONS,
                               k_means_path=model_dir)
        envs.append(environment)
    # Create the parallel envs (sequentially to prevent issues!)
    #envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # A simple funciton to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(episodes_per_thread[i]):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass
    
    evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    for thread in evaluator_threads:
        thread.start()

    # wait fo the evaluation to finish
    for thread in evaluator_threads:
        thread.join()

if __name__ == "__main__":
    main()
    

