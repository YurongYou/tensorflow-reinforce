from __future__ import print_function
from collections import deque
from rl.mace import MixtureActorCriticExperts
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import seaborn as sns
sns.set_context("paper", font_scale=1.5)

MAX_STEPS             = 50000
MAX_STEPS_PEREPISODE  = 1000
update_interval       = 1
count_episode         = 0
evaluate_interval     = 2000
NUM_EXPERTS           = 1


env_name = 'InvertedDoublePendulum-v1'
env = gym.make(env_name)

np.random.seed(123)
env.seed(123)

print("observation high & low")
print(env.observation_space.high)
print(env.observation_space.low)
print("action high & low")
print(env.action_space.high)
print(env.action_space.low)
print()

sess      = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
writer    = tf.summary.FileWriter("./experiments/{}-experiment-2".format(env_name))

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# MACE actor and critic architecture

def common_net(states):
  h1_dim = 512
  h2_dim = 256

  W1 = tf.get_variable("common_W1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("common_b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

  W2 = tf.get_variable("common_W2", [h1_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("common_b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

  return h2

def expert_action_net(common, expert_num):
  hidden = 256
  W1 = tf.get_variable("action_W1", [common.shape[1], hidden],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("action_b1", [hidden],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(common, W1) + b1)

  W2 = tf.get_variable("action_W2", [hidden, action_dim * expert_num], initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("action_b2", [action_dim * expert_num],
                        initializer=tf.constant_initializer(0))
  a = tf.tanh(tf.matmul(h1, W2) + b2)
  return tf.reshape(a, [-1, expert_num, action_dim])

def expert_value_net(common, expert_num):
  hidden = 256
  W1 = tf.get_variable("value_W1", [common.shape[1], hidden],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("value_b1", [hidden],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(common, W1) + b1)

  W2 = tf.get_variable("value_W2", [hidden, expert_num], initializer = tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("value_b2", [expert_num],
                        initializer=tf.constant_initializer(0))
  v = tf.matmul(h1, W2) + b2
  return v

mace = MixtureActorCriticExperts(sess,
                                          optimizer,
                                          state_dim,
                                          action_dim,
                                          common_net,
                                          expert_action_net,
                                          expert_value_net,
                                          expert_num=NUM_EXPERTS,
                                          exp_init_temp=20,
                                          exp_end_temp=0.025,
                                          exp_init_eta=0.9,
                                          exp_end_eta=0.2,
                                          anneal_iter=50000,
                                          summary_writer=None)



saver                 = tf.train.Saver()

state = env.reset()
done = False
step_idx = []
average_total_reward = []
for step in xrange(MAX_STEPS):
  if done :
    count_episode += 1
    state = env.reset()
  action, actor_num, isExplore = mace.sampleAction(state[np.newaxis,:])
  next_state, reward, done, _ = env.step(action)
  mace.storeExperience(state, action, reward, next_state, actor_num, isExplore, done)
  if step % update_interval == 0:
    mace.updateModel()
  state = next_state

  if step % evaluate_interval == 0:
    episode_history = deque(maxlen=100)
    print("At Step {}".format(step))
    for i_eval in range(100):
      total_rewards = 0
      state = env.reset()
      done = False
      for t in range(MAX_STEPS_PEREPISODE):
        action, actor_num, isExplore = mace.sampleAction(state[np.newaxis,:], exploration=False)
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        state = next_state
        if done: break
      episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)
    done = True
    print("Current Average Episodic Reward: {:.2f}".format(mean_rewards))
    save_path = saver.save(sess, "./experiments/{}-experiment-2/model".format(env_name), global_step=step)
    print("save model to " + save_path)
    step_idx.append(step)
    average_total_reward.append(mean_rewards)

print(step_idx)
print(average_total_reward)
with sns.axes_style("ticks"):
  plt.plot(step_idx, average_total_reward)
  plt.xlabel("steps")
  plt.ylabel("average_total_reward")
  sns.despine()
  plt.savefig("./experiments/{}-experiment-2/average_total_reward.pdf".format(env_name))