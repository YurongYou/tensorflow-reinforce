import random
import numpy as np
import tensorflow as tf
from .replay_buffer import ReplayBuffer


class MixtureActorCriticExperts(object):
    """
    Terrain-Adaptive Locomotion Skills Using Deep Reinforcement Learning
    Xue Bin Peng, Glen Berseth, Michiel van de Panne
    University of British Columbia
    """

    def __init__(self,
                    session,
                    optimizer,
                    state_dim,
                    action_dim,
                    common_net,
                    expert_action_net,
                    expert_value_net,
                    expert_num=3,
                    batch_size=32,
                    actor_buffer_size=25000,
                    critic_buffer_size=25000,
                    exp_init_temp=20,
                    exp_end_temp=0.025,
                    exp_init_eta=0.9,
                    exp_end_eta=0.2,
                    anneal_iter=50000,
                    store_replay_every=1,
                    discount_factor=0.99,
                    target_update_interval=500,
                    max_gradient=5,
                    summary_writer=None,
                    info_output_interval=1000,
                    summary_every=100):

        # tensorflow machinery
        self.session = session
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.info_output_interval = info_output_interval

        # model components
        self.common_net = common_net
        self.expert_action_net = expert_action_net
        self.expert_value_net = expert_value_net
        self.actor_buffer = ReplayBuffer(buffer_size=actor_buffer_size)
        self.critic_buffer = ReplayBuffer(buffer_size=critic_buffer_size)

        # training parameters
        self.expert_num = expert_num
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_factor = discount_factor
        self.target_update_interval = target_update_interval
        self.max_gradient = max_gradient
        self.exp_init_temp = exp_init_temp
        self.exp_end_temp = exp_end_temp
        self.exp_temp = exp_init_temp

        self.exp_init_eta = exp_init_eta
        self.exp_end_eta = exp_end_eta
        self.exp_eta = exp_init_eta

        self.anneal_iter = anneal_iter

        # counters
        self.store_replay_every = store_replay_every
        self.store_experience_cnt = 0
        self.train_iteration = 0

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            # graph was not available when journalist was created
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every

    def create_variables(self):
        with tf.name_scope("model_inputs"):
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
        with tf.name_scope("MACE"):
            with tf.variable_scope("MACE"):
                self.common = self.common_net(self.states)
                self.expertsAction = self.expert_action_net(self.common, self.expert_num)
                self.expertsValue = self.expert_value_net(self.common, self.expert_num)

        MACE_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="MACE")

        with tf.name_scope("estimate_future_rewards"):
            self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
            self.next_state_masks = tf.placeholder(tf.float32, (None,), name="next_state_masks")
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

            with tf.variable_scope("target_MACE"):
                self.target_common = self.common_net(self.next_states)
                self.target_expertsAction = self.expert_action_net(self.target_common, self.expert_num)
                self.target_expertsValue = self.expert_value_net(self.target_common, self.expert_num)

            next_action_scores = tf.reduce_max(self.target_expertsValue, reduction_indices=[1]) * self.next_state_masks
            future_rewards = self.rewards + self.discount_factor * next_action_scores

        target_MACE_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_MACE")

        with tf.name_scope("compute_gradient"):
            self.expert_chosen = tf.placeholder(tf.int32, (None, 2), name="expert_chosen")
            temp_diff =  future_rewards - tf.gather_nd(self.expertsValue, self.expert_chosen)
            # self.test = tf.gather_nd(self.expertsValue, self.expert_chosen)
            self.critic_loss = tf.reduce_mean(tf.square(temp_diff))
            self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, var_list=MACE_variables)


            for i, (grad, var) in enumerate(self.critic_gradients):
                # clip gradients by norm
                if grad is not None:
                    self.critic_gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            self.train_critic = self.optimizer.apply_gradients(self.critic_gradients)

            self.max_value_MACE = tf.reduce_max(self.expertsValue, reduction_indices=[1])
            # CACLA-like update
            self.temp_diff = future_rewards - self.max_value_MACE
            self.greater = tf.cast(tf.greater(self.temp_diff, tf.zeros(shape=tf.shape(self.temp_diff))), tf.float32)
            self.action = tf.placeholder(tf.float32, (None, self.action_dim), name="action")
            self.action_diff = tf.gather_nd(self.expertsAction, self.expert_chosen) - self.action
            self.actor_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.action_diff), axis=1) * self.greater)
            self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, var_list=MACE_variables)

            for i, (grad, var) in enumerate(self.actor_gradients):
                # clip gradients by norm
                if grad is not None:
                    self.actor_gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

            self.train_actor = self.optimizer.apply_gradients(self.actor_gradients)

        with tf.name_scope("update_target_network"):
            self.target_network_update = []

            for t, e in zip(target_MACE_variables, MACE_variables):
                update_op = tf.assign(t, e)
                self.target_network_update.append(update_op)

            # group all assignment operations together
            self.target_network_update = tf.group(*self.target_network_update)

        self.summarize = tf.summary.merge_all()
        self.no_op = tf.no_op()

    def sampleAction(self, states, exploration=True):
        assert(states.shape[0] == 1)
        actions, values = self.session.run([self.expertsAction, self.expertsValue], {self.states: states})
        isExplore = False
        if exploration:
            values = values.reshape(-1)
            values /= self.exp_temp
            values -= np.amax(values) # avoid precision issue
            values = np.exp(values)
            values /= np.sum(values)
            actor_num = np.random.choice(self.expert_num, 1, p=values)[0]
            action = actions[0, actor_num]
            isExplore = np.random.binomial(1, self.exp_eta)
            # print(isExplore == 1)
            preaction = action
            action = action + 0.1 * np.random.standard_normal(self.action_dim) * isExplore
            action = np.clip(action, -1.0, 1.0)
            # Caution! Here should not be np.clip(action, 0.0, 1.0)
            isExplore = isExplore == 1
        else:
            values = values.reshape(-1)
            actor_num = np.argmax(values)
            action = actions[0, actor_num]
        return action, actor_num, isExplore

    def updateModel(self):
        isUpdated = False

        if self.actor_buffer.count() >= self.batch_size:
            # print("actor update")
            isUpdated = True

            batch           = self.actor_buffer.getBatch(self.batch_size)
            states          = np.zeros((self.batch_size, self.state_dim))
            rewards         = np.zeros((self.batch_size,))
            actions         = np.zeros((self.batch_size, self.action_dim))
            next_states     = np.zeros((self.batch_size, self.state_dim))
            actor_nums      = np.zeros((self.batch_size, 2))
            next_state_masks = np.zeros((self.batch_size,))

            for k, (s0, a, r, s1, actor_num, done) in enumerate(batch):
                states[k]  = s0
                rewards[k] = r
                actions[k] = a
                actor_nums[k] = np.array([k, actor_num])
                if not done:
                    next_states[k] = s1
                    next_state_masks[k] = 1

            loss, _ = self.session.run([self.actor_loss, self.train_actor], {
                self.states:           states,
                self.next_states:     next_states,
                self.next_state_masks: next_state_masks,
                self.action:          actions,
                self.expert_chosen:   actor_nums,
                self.rewards:         rewards,
                })

        if self.critic_buffer.count() >= self.batch_size:
            isUpdated = True

            batch           = self.critic_buffer.getBatch(self.batch_size)
            states          = np.zeros((self.batch_size, self.state_dim))
            rewards         = np.zeros((self.batch_size,))
            actions         = np.zeros((self.batch_size, self.action_dim))
            next_states     = np.zeros((self.batch_size, self.state_dim))
            actor_nums      = np.zeros((self.batch_size, 2))
            next_state_masks = np.zeros((self.batch_size,))

            for k, (s0, a, r, s1, actor_num, done) in enumerate(batch):
                states[k]  = s0
                rewards[k] = r
                actions[k] = a
                actor_nums[k] = np.array([k, actor_num])
                if not done:
                    next_states[k] = s1
                    next_state_masks[k] = 1

            loss, _ = self.session.run([self.critic_loss, self.train_critic], {
                self.states:           states,
                self.next_states:     next_states,
                self.next_state_masks: next_state_masks,
                self.action:          actions,
                self.expert_chosen:   actor_nums,
                self.rewards:         rewards,
                })

        if isUpdated:
            self.train_iteration += 1

        if self.train_iteration % self.target_update_interval == 0:
            # print("target updated")
            self.session.run(self.target_network_update)

        self.annealExploration()

    def annealExploration(self):
        ratio = max((self.anneal_iter - self.train_iteration) / float(self.anneal_iter), 0)
        self.exp_temp = (self.exp_init_temp - self.exp_end_temp) * ratio + self.exp_end_temp

        self.exp_eta = (self.exp_init_eta - self.exp_end_eta) * ratio + self.exp_end_eta

    def storeExperience(self, state, action, reward, next_state, actor_num, isExplore, done):
        if self.store_experience_cnt % self.store_replay_every == 0 or done:
            if isExplore:
                self.actor_buffer.add(state, action, reward, next_state, actor_num, done)
            else:
                self.critic_buffer.add(state, action, reward, next_state, actor_num, done)
        self.store_experience_cnt += 1
