# Simple env test.
import json
import select
import time
import logging
import functools
import os
import tqdm
from pathlib import Path

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser
from typing import List, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical  


from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans

import coloredlogs, logging
coloredlogs.install(logging.INFO)
logger = logging.getLogger(__name__)


# Acme dependencies
import acme
import tree
from acme import specs
from acme import types
from acme.agents.tf import r2d3
from acme import wrappers
from acme.wrappers.minerl_wrapper import OVAR
from acme.wrappers import MineRLWrapper
from acme.tf import networks

import dm_env

from acme.utils import loggers
#logger = loggers.TerminalLogger(label='minerl', time_delta=10.)

# number of discrete actions
NUMBER_OF_DISCRETE_ACTIONS = 25
# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', '/hdd/minerl')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
rel_path = os.path.dirname(__file__) # relative directory path
performance_dir = os.path.join(rel_path, "performance")
Path(performance_dir).mkdir(parents=True, exist_ok=True)

parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

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
      ),
      wrappers.SinglePrecisionWrapper,
  ])

def _nested_stack(sequence: List[Any]):
  """Stack nested elements in a sequence."""
  return tree.map_structure(lambda *x: np.stack(x), *sequence)

class DemonstrationRecorder:
  """Generate (TimeStep, action) tuples
  """

  def __init__(self, environment: dm_env.Environment):
    self._demos = []
    self._environment = environment
    self.k_means = environment.k_means
    self.num_classes = self.k_means.n_clusters
    self._prev_action: types.NestedArray 
    self._prev_reward: types.NestedArray
    self._reset_episode()

  def map_action(self, action: types.NestedArray) -> types.NestedArray:
    # map from cont to discrete for the agent
    action = action['vector'].reshape(1, 64)
    action = self.k_means.predict(action)[0]
    return action

  def step(self, timestep: dm_env.TimeStep, action: np.ndarray):
    reward = np.array(timestep.reward or 0., np.float32)
    self._episode_reward += reward
    # this imitates the enviroment step to create data in the same format.
    new_timestep = self._augment_observation(timestep)
    discrete_action = self.map_action(action)
    self._prev_action = discrete_action
    self._prev_reward = reward
    return (new_timestep, discrete_action)

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    ovar = OVAR(observation=timestep.observation['pov'].astype(np.float32),
                obs_vector=timestep.observation['vector'].astype(np.float32),
                action=self._prev_action,
                reward=self._prev_reward)
    return timestep._replace(observation=ovar)

  def record_episode(self):
    logger.info(f"episode length of {len(self._episode)}")
    self._demos.append(_nested_stack(self._episode))
    self._reset_episode()

  def _reset_episode(self):
    self._episode = []
    self._episode_reward = 0
    self._prev_action = tree.map_structure(
        lambda x: x.generate_value(), self._environment.action_spec())
    self._prev_reward = tree.map_structure(
        lambda x: x.generate_value(), self._environment.reward_spec())

  @property
  def episode_reward(self):
    return self._episode_reward
  
  def _change_shape(self, shape):
      shape = list(shape)
      shape[0] = None
      return tuple(shape)
  
  def _change_type(self, _type):
    if _type == np.dtype('float64'):
      return np.dtype('float32')
    else:
      return _type

  def make_tf_dataset(self):
    types = tree.map_structure(lambda x: self._change_type(x.dtype), self._demos[0])
    shapes = tree.map_structure(lambda x: self._change_shape(x.shape), self._demos[0])
    logger.info({"types": types})
    ds = tf.data.Dataset.from_generator(lambda: self._demos, types, shapes)
    return ds.repeat().shuffle(len(self._demos))

def generate_demonstration(env: dm_env.Environment, 
                         dat_loader: minerl.data.data_pipeline.DataPipeline, 
                         nb_experts: int = 20):
  # Build demonstrations.    
  recorder = DemonstrationRecorder(env)
  recorder._reset_episode()
  # replay trajectories
  trajectories = dat_loader.get_trajectory_names()
  t = 0
  for t, trajectory in enumerate(trajectories):
      if t < nb_experts:
        logger.info({str(t): trajectory})
        for i, (state, a, r, _, done, meta) in enumerate(dat_loader.load_data(trajectory, include_metadata=True)):    
          if done:
            step_type = dm_env.StepType(2)
          elif i == 0:
            step_type = dm_env.StepType(0)
          else:
            step_type = dm_env.StepType(1)
          ts = dm_env.TimeStep(observation=state, 
                               reward=r, 
                               step_type=step_type, 
                               discount=np.array(1., dtype=np.float32))
          yield recorder.step(ts, a)

def main():
    """
    This function will be called for training phase.
    """
  
    rel_path = os.path.dirname(__file__) # relative directory path
    model_dir = os.path.join(rel_path, "train")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    burn_in_length = 40
    trace_length = 40

    # Create data loader
    logger.info((MINERL_GYM_ENV, MINERL_DATA_ROOT))
    data = minerl.data.make(MINERL_GYM_ENV, 
                            data_dir=MINERL_DATA_ROOT, 
                            num_workers=1,
                            worker_batch_size=4)

    # Create env
    logger.info("creating environment")
    environment = make_environment(num_actions=NUMBER_OF_DISCRETE_ACTIONS,
                                   k_means_path=model_dir,
                                   dat_loader=data)
    spec = specs.make_environment_spec(environment)

    # # Create a logger for the agent and environment loop.
    # agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
    # env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

    # Build demonstrations
    logger.info("building the demonstration dataset")
    generator = generate_demonstration(environment, data)
    logger.info("demonstration dataset is built..")

    # Construct the network.
    network = create_network()
    target_network = create_network()

    logger.info(f"model directory: {model_dir}")
    # sequence_length = burn_in_length + trace_length
    agent = r2d3.R2D3(
        model_directory=model_dir,
        environment_spec=spec,
        network=network,
        target_network=target_network,
        demonstration_generator=generator,
        demonstration_ratio=0.5,
        batch_size=8,
        samples_per_insert=2,
        min_replay_size=1000,
        max_replay_size=100_000,
        burn_in_length=burn_in_length,
        trace_length=trace_length,
        replay_period=40, # per R2D3 paper.
        checkpoint=True,
        #logger=agent_logger
    )

    # Run the env loop
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_steps=MINERL_TRAINING_MAX_STEPS)  # pytype: disable=attribute-error

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    environment.close()


if __name__ == "__main__":
  main()
