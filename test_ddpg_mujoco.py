from __future__ import print_function
from collections import deque
from rl.pg_ddpg import DeepDeterministicPolicyGradient
import tensorflow as tf
import numpy as np
import gym
import os

env_name = 'InvertedDoublePendulum-v1'
# env_name = 'Hopper-v1'
env = gym.make(env_name)


np.random.seed(123)
env.seed(123)

MAX_STEPS             = 50000
MAX_STEPS_PEREPISODE  = 1000
update_interval       = 1
count_episode         = 0
evaluate_interval     = 2000

sess      = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("observation high & low")
print(env.observation_space.high)
print(env.observation_space.low)
print("action high & low")
print(env.action_space.high)
print(env.action_space.low)
print()

if not os.path.exists("./experiments/{}-experiment-ddpg/".format(env_name)):
  os.mkdir("./experiments/{}-experiment-ddpg/".format(env_name))

# DDPG actor and critic architecture
# Continuous control with deep reinforcement learning
# Timothy P. Lillicrap, et al., 2015

def actor_network(states):
  h1_dim = 400
  h2_dim = 300

  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

  W2 = tf.get_variable("W2", [h1_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

  # use tanh to bound the action
  W3 = tf.get_variable("W3", [h2_dim, action_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.get_variable("b3", [action_dim],
                       initializer=tf.constant_initializer(0))

  # we assume actions range from [-1, 1]
  # you can scale action outputs with any constant here
  a = tf.nn.tanh(tf.matmul(h2, W3) + b3)
  return a

def critic_network(states, action):
  h1_dim = 400
  h2_dim = 300

  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
  # skip action from the first layer
  h1_concat = tf.concat(axis=1, values=[h1, action])

  W2 = tf.get_variable("W2", [h1_dim + action_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1_concat, W2) + b2)

  W3 = tf.get_variable("W3", [h2_dim, 1],
                       initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.get_variable("b3", [1],
                       initializer=tf.constant_initializer(0))
  v = tf.matmul(h2, W3) + b3
  return v

pg_ddpg = DeepDeterministicPolicyGradient(sess,
                                          optimizer,
                                          actor_network,
                                          critic_network,
                                          state_dim,
                                          action_dim,
                                          summary_writer=None)
saver = tf.train.Saver()

for kase in xrange(5):
  print("======== test case No.{} ========".format(kase))
  step_idx = []
  average_total_reward = []
  var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
  sess.run(tf.variables_initializer(var_lists))
  done = False
  state = env.reset()
  for step in xrange(MAX_STEPS):
    if done :
      count_episode += 1
      state = env.reset()
    action = pg_ddpg.sampleAction(state[np.newaxis,:])
    next_state, reward, done, _ = env.step(action[0])
    pg_ddpg.updateModel()
    state = next_state

    if step % evaluate_interval == 0:
      episode_history = deque(maxlen=100)
      print("At Step {}".format(step))
      for i_eval in range(100):
        total_rewards = 0
        state = env.reset()
        done = False
        for t in range(MAX_STEPS_PEREPISODE):
          action = pg_ddpg.sampleAction(state[np.newaxis,:], exploration=False)
          next_state, reward, done, _ = env.step(action[0])
          total_rewards += reward
          state = next_state
          if done: break
        episode_history.append(total_rewards)
      mean_rewards = np.mean(episode_history)
      done = True
      print("Current Average Episodic Reward: {:.2f}".format(mean_rewards))
      step_idx.append(step)
      average_total_reward.append(mean_rewards)
  save_path = saver.save(sess, "./experiments/{}-experiment-ddpg/model{}".format(env_name, kase))
  print("save model to " + save_path)
  print(step_idx)
  print(average_total_reward)