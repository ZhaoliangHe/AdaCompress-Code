# encoding: utf-8
import numpy as np
import tensorflow as tf
from collections import deque

np.random.seed(1)
tf.set_random_seed(1)


class PG_Agent(object):
    def __init__(self,
                 n_features,
                 n_actions,
                 epoch_per_episode=1,
                 init_learning_rate=0.01,
                 reward_decay=0.95,
                 greedy_epsilon=0.8,
                 q_value_decay=0.9):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epoch_per_episode = epoch_per_episode
        self.init_lr = init_learning_rate
        self.gamma = reward_decay
        self.greedy_epsilon = greedy_epsilon
        self.q_value_decay = q_value_decay

        self.history_observations = []
        self.history_actions = []
        self.history_rewards = []

        self._build_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_observations = tf.placeholder(tf.float32, [None, self.n_features], name='observations')
            self.tf_actions = tf.placeholder(tf.float32, [None, self.n_actions], name='actions')
            self.tf_ac_reward = tf.placeholder(tf.float32, [None, ], name='actions_reward')

        # fc1
        fc1 = tf.layers.dense(
            inputs=self.tf_observations,
            units=256,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        fc2 = tf.layers.dense(
            inputs=fc1,
            units=256,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.fc3 = tf.layers.dense(
            inputs=fc2,
            units=self.n_actions,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )

        self.all_act_prob = tf.layers.dense(
            inputs=self.fc3,
            units=self.n_actions,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='output'
        )

        with tf.name_scope('loss'):
            # -log(prob) * vt
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # loss = tf.reduce_sum(-tf.log(tf.reduce_mean(self.all_act_prob))) * self.tf_ac_reward
            # 真实行为与我的很不一样，却获得了大reward，意味着我要进行大更新
            self.loss = tf.reduce_sum(
                tf.exp(tf.reduce_mean(tf.square(self.all_act_prob - self.tf_actions))) * self.tf_ac_reward)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.init_lr).minimize(self.loss)

    def inference(self, observation):
        policy_network_output = self.sess.run(self.all_act_prob,
                                              feed_dict={self.tf_observations: np.expand_dims(observation, axis=0)})[0]
        # action_out = \
        # np.array([np.random.choice([0., 1.], size=1, p=[1 - item, item]) for item in policy_network_output])[..., 0]
        return policy_network_output

    def remember(self, observation, action, reward):
        self.history_observations.append(observation)
        self.history_actions.append(action)
        self.history_rewards.append(reward)

    def _discount_norm_reward(self):
        discounted_rewards = np.zeros_like(self.history_rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, len(self.history_rewards))):
            running_add = running_add * self.gamma + self.history_rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards

    def replay(self):
        discounted_norm_reward = self._discount_norm_reward()

        for i in range(self.epoch_per_episode):
            self.sess.run(self.train_op, feed_dict={
                self.tf_observations: np.vstack(self.history_observations),
                self.tf_actions: np.vstack(self.history_actions),
                self.tf_ac_reward: discounted_norm_reward
            })

        self.history_observations, self.history_actions, self.history_rewards = [], [], []
        return discounted_norm_reward


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.index = 0

        self.states = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.new_states = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)

    def remember(self, s, a, r, s_):
        self.states += [s]
        self.actions += [a]
        self.new_states += [s_]
        self.rewards += [r]

        self.index += len(s)


class DDPG_Agent(object):
    def __init__(self, a_dim, s_dim, train_batchsize):
        # hyper parameters
        self.LR_A = 0.01  # learning rate for actor
        self.LR_C = 0.01  # learning rate for critic
        self.GAMMA = 0.9  # reward discount
        self.TAU = 0.03  # soft replacement
        self.MEMORY_CAPACITY = 20000
        self.BATCH_SIZE = train_batchsize

        self.memory = Memory(capacity=self.MEMORY_CAPACITY)
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.actor_eval_net = self._build_a(self.S, scope='eval', trainable=True)
            actor_target_net = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.critic_eval_net = self._build_c(self.S, self.actor_eval_net, scope='eval', trainable=True)
            self.critic_target_net = self._build_c(self.S_, actor_target_net, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + self.GAMMA * self.critic_target_net
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.critic_eval_net)
        self.ctrain = tf.train.AdamOptimizer(self.LR_C).minimize(self.td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.critic_eval_net)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.actor_eval_net, {self.S: s})

    def learn(self, epochs=5):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(len(self.memory.actions), size=self.BATCH_SIZE)

        batch_states = np.vstack(self.memory.states)[indices]
        batch_actions = np.vstack(self.memory.actions)[indices]
        batch_rewards = np.vstack(self.memory.rewards)[indices]
        batch_newstates = np.vstack(self.memory.new_states)[indices]

        for i in range(epochs):
            self.sess.run(self.atrain, {self.S: batch_states})
            self.sess.run(self.ctrain, {self.S: batch_states, self.actor_eval_net: batch_actions, self.R: batch_rewards,
                                        self.S_: batch_newstates})

            # print(
            #     "\t\tQ: %.3f\ttd_error: %.3f" % (np.mean(self.sess.run(self.critic_eval_net,
            #                                                            feed_dict={
            #                                                                self.S: batch_states,
            #                                                                self.actor_eval_net: batch_actions})),
            #                                      np.mean(self.sess.run(self.td_error,
            #                                                            feed_dict={self.S: batch_states,
            #                                                                       self.S_: batch_newstates,
            #                                                                       self.actor_eval_net: batch_actions,
            #                                                                       self.R: batch_rewards}))))

        self.memory.index = 0

    def store_transition(self, s, a, r, s_):
        self.memory.remember(s, a, np.vstack([r] * len(s)), s_)
        self.memory.index += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu, name='fc1', trainable=trainable)
            fc2 = tf.layers.dense(inputs=fc1, units=64, activation=tf.nn.relu, name='fc2', trainable=trainable)
            a = tf.layers.dense(inputs=fc2, units=self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            s_fc1 = tf.layers.dense(inputs=s, units=128, activation=tf.nn.relu, name='s_fc1', trainable=trainable)
            s_fc2 = tf.layers.dense(inputs=s_fc1, units=64, activation=tf.nn.relu, name='s_fc2', trainable=trainable)
            s_out = tf.layers.dense(inputs=s_fc2, units=32, activation=tf.nn.relu, name='s_out', trainable=trainable)

            a_fc1 = tf.layers.dense(inputs=a, units=64, activation=tf.nn.relu, name='a_fc1', trainable=trainable)
            a_out = tf.layers.dense(inputs=a_fc1, units=32, activation=tf.nn.relu, name='a_out', trainable=trainable)

            n_l1 = 16
            w1_s = tf.get_variable('w1_s', [32, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [32, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s_out, w1_s) + tf.matmul(a_out, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
