# Simple env test.
import json
import select
import time
import logging
import functools
import os
import tqdm

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.INFO)


# Acme dependencies
import acme
from acme import specs
from acme.agents.tf import r2d3
from acme import wrappers
from acme.tf import networks
from acme.agents.tf.dqfd import bsuite_demonstrations

import dm_env

from acme.utils import loggers
logger = loggers.TerminalLogger(label='minerl', time_delta=10.)

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
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
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


def make_environment(num_actions: int, dat_loader: minerl.data.data_pipeline.DataPipeline,) -> dm_env.Environment:
  """
  Wrap the environment in:
    1 - MineRLWrapper 
        - similar to OAR but add proprioceptive features
        - kMeans to map cont action space to a discrete one
    2 - SinglePrecisionWrapper
    3 - GymWrapper
  """

  env = gym.make(MINERL_GYM_ENV)
      
  return wrappers.wrap_all(env, [
      wrappers.GymWrapper,
      functools.partial(
        wrappers.MineRLWrapper,
          num_actions=num_actions,
          dat_loader=dat_loader,
      ),
      wrappers.SinglePrecisionWrapper,
  ])

def _nested_stack(sequence: List[Any]):
  """Stack nested elements in a sequence."""
  return tree.map_structure(lambda *x: np.stack(x), *sequence)

class DemonstrationRecorder:
  """Records demonstrations.

  A demonstration is a (observation, action, reward, discount) tuple where
  every element is a numpy array corresponding to a full episode.
  """

  def __init__(self):
    self._demos = []
    self._reset_episode()

  def step(self, timestep: dm_env.TimeStep, action: np.ndarray):
    reward = np.array(timestep.reward or 0, np.float32)
    self._episode_reward += reward
    self._episode.append((timestep.observation, action, reward,
                          np.array(timestep.discount or 0, np.float32)))

  def record_episode(self):
    self._demos.append(_nested_stack(self._episode))
    self._reset_episode()

  def discard_episode(self):
    self._reset_episode()

  def _reset_episode(self):
    self._episode = []
    self._episode_reward = 0

  @property
  def episode_reward(self):
    return self._episode_reward
  
  def _change_shape(self, shape):
      shape = list(shape)
      shape[0] = None
      return tuple(shape)

  def make_tf_dataset(self):
    types = tree.map_structure(lambda x: x.dtype, self._demos[0])
    shapes = tree.map_structure(lambda x: self._change_shape(x.shape), self._demos[0])
    ds = tf.data.Dataset.from_generator(lambda: self._demos, types, shapes)
    return ds.repeat().shuffle(len(self._demos))

def build_demonstrations(dat_loader: minerl.data.data_pipeline.DataPipeline, sequence_length: int, nb_experts: int = 5):
  # Build demonstrations.    
  recorder = DemonstrationRecorder()
  # replay trajectories
  trajectories = dat_loader.get_trajectory_names()
  for t, trajectory in enumerate(trajectories):
    if t < nb_experts:
      logger.write({str(t): trajectory})
      for i, (state, a, r, next_state, done, meta) in enumerate(dat_loader.load_data(trajectory, include_metadata=True)):
        
        if done:
          step_type = dm_env.StepType(2)
        elif i == 0:
          step_type = dm_env.StepType(0)
        else:
          step_type = dm_env.StepType(1)
        ts = dm_env.TimeStep(observation=state, reward=r, step_type=step_type, discount=0)
        recorder.step(ts, a)
      recorder.record_episode()
  return recorder.make_tf_dataset()

def main():
    """
    This function will be called for training phase.
    """
    burn_in_length = 40
    trace_length = 40
    sequence_length = burn_in_length + trace_length + 1 #per R2D3 agent

    # Create data loader
    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    # Create env
    environment = make_environment(num_actions=30, dat_loader=data)
    spec = specs.make_environment_spec(environment)

    # Create a logger for the agent and environment loop.
    agent_logger = loggers.TerminalLogger(label='agent', time_delta=10.)
    env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10.)

    # Build demonstrations
    demonstration_dataset = build_demonstrations(data, sequence_length)

    # Construct the network.
    network = create_network()
    target_network = create_network()

    # sequence_length = burn_in_length + trace_length
    agent = r2d3.R2D3(
        environment_spec=spec,
        network=network,
        target_network=target_network,
        demonstration_dataset=demonstration_dataset,
        demonstration_ratio=0.5,
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10,
        burn_in_length=burn_in_length,
        trace_length=trace_length,
        replay_period=40, # per R2D3 paper.
        checkpoint=False,
        logger=agent_logger
    )

    # Run the env loop
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_steps=MINERL_TRAINING_MAX_STEPS)  # pytype: disable=attribute-error

    # Save trained model to train/ directory
    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    #env.close()


if __name__ == "__main__":
    main()
