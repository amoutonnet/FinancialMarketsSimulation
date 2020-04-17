import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import random
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools
from . import utils
np.set_printoptions(suppress=True, linewidth=1000, threshold=float('inf'))


DELTA = 1e-10
VAR = 1
SIGMA = 1
LOG_SIGMA = 0
NPY_SQRT1_2 = 1 / (2**0.5)
NPY_PI = np.pi


def normalize(x):
    x -= x.mean()
    x /= (x.std() + DELTA)
    return x


class RLAgent():
    def __init__(self, obs_space_shape, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers, hidden_dense_layers, verbose):
        self.obs_space_shape = obs_space_shape  # The shape of the observation space
        self.gamma = gamma  # The discount rate associated with the agent experience
        self.alpha = alpha  # The learning rate for the parameter of the policy network
        self.beta = beta  # The learning rate for the parameter of the value network
        self.temp = temp  # The temperature parameter for the entropy term
        self.lambd = lambd  # The lambda parameter for general advantage estimate
        self.epsilon = epsilon  # Epsilon for the clipped loss of PPO
        self.optimizer_actor = tf.keras.optimizers.Adam(self.alpha)  # The optimizert for the actor
        self.optimizer_critic = tf.keras.optimizers.Adam(self.beta)  # The optimizert for the critic
        self.hidden_conv_layers = hidden_conv_layers  # The characteristics of the hidden convolutionnal layers
        self.hidden_dense_layers = hidden_dense_layers  # The characteristics of the hidden dense layers
        self.loss_actor = float('inf')  # The loss of the actor
        self.loss_critic = -float('inf')  # The loss of the critic
        self.memory = list()   # The memory to track trajectories
        self.verbose = verbose   # Whether you want to print things or not
        self.name = None   # Name of the network

    def get_base_network(self, obs, agent_name, network_name, initializer='random_normal'):
        x = obs
        for id_, c in enumerate(self.hidden_conv_layers):
            x = tf.keras.layers.Conv1D(filters=c[0], kernel_size=c[1], padding='same', activation='relu',
                                       kernel_initializer=initializer,
                                       bias_initializer=initializer,
                                       name='%s_%s_conv_%d' % (agent_name, network_name, id_))(x)
        x = tf.keras.layers.Flatten(name='%s_%s_flatten' % (agent_name, network_name))(x)
        for id_, d in enumerate(self.hidden_dense_layers):
            x = tf.keras.layers.Dense(d, activation='relu',
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer,
                                      name='%s_%s_dense_%d' % (agent_name, network_name, id_))(x)
        return x

    def build_network(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def get_actions(self, observations):
        raise NotImplementedError

    def get_advantages(self, critic_values, rewards):
        values = np.empty((len(rewards),))
        for i in range(len(rewards) - 1):
            values[i] = rewards[i] + self.gamma * critic_values[i + 1] - critic_values[i]
        values[-1] = rewards[-1] - critic_values[-1]
        return np.array(list(itertools.accumulate(values[::-1], lambda x, y: x * (self.gamma * self.lambd) + y))[::-1], dtype=np.float32)

    def get_discounted_rewards(self, rewards):
        return np.array(list(itertools.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float32)

    def print_verbose(self, ep, total_episodes, episode_reward, rolling_score):
        if self.verbose:
            print('{} \
                 | Current Score ({:3.2f}) Rolling Average ({:3.2f}) \
                 | Actor Loss ({:.4f}) Critic Loss ({:.4f})'.format(self.name.ljust(10),
                                                                    episode_reward,
                                                                    rolling_score,
                                                                    self.loss_actor,
                                                                    self.loss_critic
                                                                    )
                  )


class MarketMakerRL(RLAgent):
    def __init__(self, Nm, obs_space_shape, action_space_limits, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers=[], hidden_dense_layers=[128], verbose=False):
        super().__init__(obs_space_shape, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers, hidden_dense_layers, verbose)
        self.action_space_lower_limit = action_space_limits[0]
        self.action_space_upper_limit = action_space_limits[1]
        self.Nm = Nm
        self.name = 'Market Makers'
        self.build_network()

    def build_network(self):
        """
        This function is used to build the neural network of the agent
        """
        obs = tf.keras.Input(shape=self.obs_space_shape, name='market_makers_obs')
        advantages = tf.keras.Input(shape=(1,), name='market_makers_advantages')
        x = self.get_base_network(obs, 'market_makers', 'actor', 'random_normal')
        mu = tf.keras.layers.Dense(1, activation='linear',
                                   kernel_initializer='random_normal',
                                   name='market_makers_means')(x)
        self.policy = tf.keras.Model(inputs=obs, outputs=mu, name='Market_Makers_Policy')
        self.actor = tf.keras.Model(inputs=[obs, advantages], outputs=mu, name='Market_Makers_Actor')

        def actor_loss(y_true, y_pred):
            mu = y_pred
            action = y_true

            def cdf_gauss(a):
                x = a * NPY_SQRT1_2
                z = K.abs(x)
                half_erfc_z = 0.5 * tf.math.erf(z)
                return tf.where(
                    z < NPY_SQRT1_2,
                    0.5 + 0.5 * tf.math.erf(x),
                    tf.where(
                        x > 0,
                        1.0 - half_erfc_z,
                        half_erfc_z
                    )
                )

            def log_cdf_gauss(x):
                def safe_log(x):
                    return K.log(tf.where(x > 0, x, DELTA))

                return tf.where(
                    x > 6,
                    -cdf_gauss(-x),
                    tf.where(
                        x > -14,
                        safe_log(cdf_gauss(x)),
                        -0.5 * K.square(x) - safe_log(-x) - 0.5 * K.log(2 * NPY_PI)
                    )
                )

            log_lik = tf.where(
                action < self.action_space_upper_limit,
                tf.where(
                    action > self.action_space_lower_limit,
                    -0.5 * K.log(2 * NPY_PI) - LOG_SIGMA - 0.5 * K.square(action - mu) / VAR,
                    log_cdf_gauss((self.action_space_lower_limit - mu) / SIGMA)
                ),
                log_cdf_gauss(-(self.action_space_upper_limit - mu) / SIGMA)
            )
            old_log_lik = K.stop_gradient(log_lik)
            advantages_with_entropy = advantages - self.temp * old_log_lik
            ratio = K.exp(log_lik - old_log_lik)
            clipped_ratio = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            return -K.sum(K.minimum(ratio * advantages_with_entropy, clipped_ratio * advantages_with_entropy), keepdims=True) / self.Nm

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

        x = self.get_base_network(obs, 'market_makers', 'critic', 'random_normal')

        values = tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer='random_normal',
                                       name='market_makers_values')(x)

        self.critic = tf.keras.Model(inputs=obs, outputs=values, name='Market_Makers_Critic')
        self.critic.compile(loss='mse', optimizer=self.optimizer_critic, experimental_run_tf_function=False)

        if self.verbose:
            utils.print_to_output(title='Market Makers Networks')
            self.actor.summary()
            print()
            self.critic.summary()

    def get_actions(self, observations):
        means = self.policy(observations)
        actions = np.random.normal(means, SIGMA, len(means))
        clipped_actions = np.clip(actions, self.action_space_lower_limit, self.action_space_upper_limit)
        return clipped_actions

    def learn(self, epochs=1, batch_size=8):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        observations, actions, rewards = map(np.array, zip(*self.memory))
        self.loss_actor = 0
        self.loss_critic = 0
        for i in range(observations.shape[1]):
            # We process the states values with the critic network
            critic_values = np.squeeze(self.critic(observations[:, i]).numpy())
            # We get the advantage (difference between the discounted reward and the baseline)
            advantages = self.get_advantages(critic_values, rewards[:, i])
            # We normalize advantages
            advantages = normalize(advantages)
            # We train the actor network
            self.loss_actor += self.actor.train_on_batch([observations[:, i], advantages], actions[:, i])
            # We get discounted rewards
            discounted_rewards = self.get_discounted_rewards(rewards[:, i])
            # We train the critic network
            self.loss_critic += self.critic.train_on_batch(observations[:, i], discounted_rewards)
        self.memory.clear()


class DealerRL(RLAgent):
    def __init__(self, Nd, obs_space_shape, action_space_shape, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers=[], hidden_dense_layers=[128], verbose=False):
        super().__init__(obs_space_shape, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers, hidden_dense_layers, verbose)
        self.action_space_shape = action_space_shape
        self.max_amount = int((action_space_shape[1] - 1) / 2)
        self.name = 'Dealers'
        self.Nd = Nd
        self.build_network()

    def build_network(self):
        """
        This function is used to build the neural network of the agent
        """
        obs = tf.keras.Input(shape=self.obs_space_shape, name="dealers_obs")
        advantages = tf.keras.Input(shape=(1,), name="dealers_advantages")
        x = self.get_base_network(obs, 'dealers', 'actor', 'random_normal')
        probability_distrib_parameters = [
            tf.keras.layers.Dense(self.action_space_shape[1], activation='softmax',
                                  kernel_initializer='random_normal',
                                  name='dealers_probs_%d' % id_)(x)
            for id_ in range(self.action_space_shape[0])
        ]
        self.policy = tf.keras.Model(inputs=obs, outputs=probability_distrib_parameters, name='Dealers_Policy')
        self.actor = tf.keras.Model(inputs=[obs, advantages], outputs=probability_distrib_parameters, name='Dealers_Actor')

        def actor_loss(y_true, y_pred):
            out = K.clip(y_pred, DELTA, 1)
            log_lik = y_true * K.log(out)
            old_log_lik = K.stop_gradient(log_lik)
            advantages_with_entropy = advantages - self.temp * K.sum(old_log_lik, axis=-1)
            ratio = K.sum(K.exp(log_lik - old_log_lik), axis=1)
            clipped_ratio = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)
            return -K.sum(K.minimum(ratio * advantages_with_entropy, clipped_ratio * advantages_with_entropy), keepdims=True) / self.Nd

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

        x = self.get_base_network(obs, 'dealers', 'critic', 'random_normal')
        values = tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer='random_normal',
                                       name="dealers_values")(x)

        self.critic = tf.keras.Model(inputs=obs, outputs=values, name='Dealers_Critic')
        self.critic.compile(loss='mse', optimizer=self.optimizer_critic, experimental_run_tf_function=False)

        if self.verbose:
            utils.print_to_output(title='Dealers Networks')
            self.actor.summary()
            print()
            self.critic.summary()

    def get_actions(self, observations):
        probabilities = K.stack(self.policy(observations), axis=1).numpy()
        print(probabilities)
        cumsums = np.cumsum(probabilities, axis=-1)
        unif_draws = np.random.rand(cumsums.shape[0], cumsums.shape[1], 1)
        actions = (unif_draws < cumsums).argmax(axis=-1)
        print(actions)
        return actions - self.max_amount

    def learn(self, epochs=1, batch_size=8):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        observations, actions, rewards = map(np.array, zip(*self.memory))
        # We rescale actions
        actions += self.max_amount
        self.loss_actor = 0
        self.loss_critic = 0
        for i in range(observations.shape[1]):
            # We process the states values with the critic network
            critic_values = np.squeeze(self.critic(observations[:, i]).numpy())
            # We one-hot encode actions
            actions_list = [tf.keras.utils.to_categorical(actions[:, i, id_], num_classes=self.action_space_shape[1]) for id_ in range(self.action_space_shape[0])]
            # We get the advantage (difference between the discounted reward and the baseline)
            advantages = self.get_advantages(critic_values, rewards[:, i])
            # We normalize advantages
            advantages = normalize(advantages)
            # We train the actor network
            self.loss_actor += self.actor.train_on_batch([observations[:, i], advantages], actions_list)[0]
            # We get discounted rewards
            discounted_rewards = self.get_discounted_rewards(rewards[:, i])
            # We train the critic network
            self.loss_critic += self.critic.train_on_batch(observations[:, i], discounted_rewards)
        self.memory.clear()
