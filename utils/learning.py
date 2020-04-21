import numpy as np
import random
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import itertools
from . import utils
np.set_printoptions(suppress=True, linewidth=1000)


DELTA = utils.DELTA
VAR = 1
SIGMA = 1
LOG_SIGMA = 0
NPY_SQRT1_2 = 1 / (2**0.5)
NPY_PI = np.pi

<<<<<<< HEAD
    Here are the descriptions of the rows:
    - The 1st one is the ask price for the time step
    - The 2nd one is the portfolio of the market maker (what was available to sell at the beginning of the time step)
    - The 3rd one is the number of sales processed during the time step
    - The 4th one is the number of purchases during the time step
    """
    # TO COMPLETE
    reward = None
    #print(observation)
    #sys.exit()
    return reward
=======

class RLAgent():
    def __init__(self, N, obs_space_shape, gamma, alpha, beta, temp, lambd, epsilon, hidden_conv_layers, hidden_dense_layers, initializer, verbose):
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
        self.initializer = initializer  # The initializer for main part of network
        self.loss_actor = [float('inf')] * N  # The loss of the actor for each agent
        self.loss_critic = [-float('inf')] * N  # The loss of the critic for each agent
        self.memory = list()   # The memory to track trajectories
        self.verbose = verbose   # Whether you want to print things or not
        self.name = None   # Name of the network
        self.N = N   # The number of agent handled by the network
>>>>>>> c6676f9092e4e2cfe890a4617625f092b76870e5

    def get_base_network(self, obs, agent_name, network_name):
        x = obs
        for id_, c in enumerate(self.hidden_conv_layers):
            x = tf.keras.layers.Conv1D(filters=c[0], kernel_size=c[1], padding='same', activation='relu',
                                       kernel_initializer=self.initializer,
                                       bias_initializer=self.initializer,
                                       name='%s_%s_conv_%d' % (agent_name, network_name, id_))(x)
        x = tf.keras.layers.Flatten(name='%s_%s_flatten' % (agent_name, network_name))(x)
        for id_, d in enumerate(self.hidden_dense_layers):
            x = tf.keras.layers.Dense(d, activation='relu',
                                      kernel_initializer=self.initializer,
                                      bias_initializer=self.initializer,
                                      name='%s_%s_dense_%d' % (agent_name, network_name, id_))(x)
        return x

<<<<<<< HEAD
def get_reward_dealers(observation):
    """
    observation is an array of size (3*nb_companies + 7, window_size + 1)
    Rows represent the state of some caracteristics during the window_size last time steps plus the current one.
    Here the current time steps is not fully updated as the training phase occurs when the step is not finished.
    Only the state of the market (1st 2*nb_companies rows), the portfolio value and the global wealth is updated.
    Therefore you should only consider the first window_size columns as the last state, the last column will be used
    to take a decision according to the current state of the market.
=======
    def build_network(self):
        raise NotImplementedError
>>>>>>> c6676f9092e4e2cfe890a4617625f092b76870e5

    def learn(self):
        raise NotImplementedError

    def get_actions(self, observations, nbs_simple_policy):
        raise NotImplementedError

    def get_action_simple_policy(self, observation):
        raise NotImplementedError

    def get_advantages(self, critic_values, rewards):
        values = np.empty((len(rewards),))
        for i in range(len(rewards) - 1):
            values[i] = rewards[i] + self.gamma * critic_values[i + 1] - critic_values[i]
        values[-1] = rewards[-1] - critic_values[-1]
        return np.array(list(itertools.accumulate(values[::-1], lambda x, y: x * (self.gamma * self.lambd) + y))[::-1], dtype=np.float32)

    def get_discounted_rewards(self, rewards):
        return np.array(list(itertools.accumulate(rewards[::-1], lambda x, y: x * self.gamma + y))[::-1], dtype=np.float32)

    def print_verbose(self, episode_reward, rolling_score):
        if self.verbose:
            for i in range(self.N):
                print('Current Score ({:3.2f}) Rolling Average ({:3.2f}) \
                    | Actor Loss ({:.4f}) Critic Loss ({:.4f})'.format(episode_reward[i],
                                                                       rolling_score[i],
                                                                       self.loss_actor[i],
                                                                       self.loss_critic[i]
                                                                       ))


class MarketMakerRL(RLAgent):
    def __init__(
        self,
        N,
        obs_space_shape,
        action_space_limits,
        gamma,
        alpha,
        beta,
        temp,
        lambd,
        epsilon,
        hidden_conv_layers=[],
        hidden_dense_layers=[128],
        initializer='random_normal',
        verbose=False
    ):
        super().__init__(
            N,
            obs_space_shape,
            gamma,
            alpha,
            beta,
            temp,
            lambd,
            epsilon,
            hidden_conv_layers,
            hidden_dense_layers,
            initializer,
            verbose
        )
        self.action_space_lower_limit = action_space_limits[0]
        self.action_space_upper_limit = action_space_limits[1]
        self.name = 'Market Makers'
        self.build_network()

    def build_network(self):
        """
        This function is used to build the neural network of the agent
        """
        obs = tf.keras.Input(shape=self.obs_space_shape, name='market_makers_obs')
        advantages = tf.keras.Input(shape=(1,), name='market_makers_advantages')
        x = self.get_base_network(obs, 'market_makers', 'actor')
        mu = tf.keras.layers.Dense(1, activation='linear',
                                   kernel_initializer=self.initializer,
                                   bias_initializer=self.initializer,
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
            return -K.mean(K.minimum(ratio * advantages_with_entropy, clipped_ratio * advantages_with_entropy), keepdims=True) / self.N

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

        x = self.get_base_network(obs, 'market_makers', 'critic')

        values = tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer=self.initializer,
                                       bias_initializer=self.initializer,
                                       name='market_makers_values')(x)

        self.critic = tf.keras.Model(inputs=obs, outputs=values, name='Market_Makers_Critic')
        self.critic.compile(loss='mse', optimizer=self.optimizer_critic, experimental_run_tf_function=False)

        if self.verbose:
            utils.print_to_output(title='Market Makers Networks')
            self.actor.summary()
            print()
            self.critic.summary()

    def get_action_simple_policy(self, observation):
        """This function takes as input an observation of one market maker
        and output a mean for the new ask price to be drawn. This ask price will be
        drawn according to a normal distribution of stddev SIGMA=1 and mean what
        this function returns.

        Arguments:
            observation {[np.ndarray]} -- [The market maker observation array described in the report]

        Returns:
            action {[int]} -- [the mean of the ask price to be drawn]
        """
        print(observation)
        # sys.exit()  # Uncomment this line to stop the program just after the print
        # TO COMPLETE (replace 10)
        action = 10
        assert(action >= self.action_space_lower_limit and action <= self.action_space_upper_limit)
        return [action]

    def get_actions(self, observations, nbs_simple_policy):
        if nbs_simple_policy:
            if nbs_simple_policy < len(observations):
                means = self.policy(observations[:-nbs_simple_policy])
                simple_policy_means = np.array([self.get_action_simple_policy(obs) for obs in observations[-nbs_simple_policy:]], dtype=np.float32)
                means = np.vstack((means, simple_policy_means))
            else:
                means = np.array([self.get_action_simple_policy(obs) for obs in observations], dtype=np.float32)
        else:
            means = self.policy(observations)
        actions = np.random.normal(means, SIGMA, len(means))
        clipped_actions = np.clip(actions, self.action_space_lower_limit, self.action_space_upper_limit)
        return clipped_actions

    def learn(self, epochs=1, batch_size=8):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        observations, actions, rewards = map(np.array, zip(*self.memory))
        for i in range(observations.shape[1]):
            # We process the states values with the critic network
            critic_values = np.squeeze(self.critic(observations[:, i]).numpy())
            # We get the advantage (difference between the discounted reward and the baseline)
            advantages = self.get_advantages(critic_values, rewards[:, i])
            # We normalize advantages
            advantages = utils.normalize(advantages)
            # We train the actor network
            self.loss_actor[i] = self.actor.train_on_batch([observations[:, i], advantages], actions[:, i])
            # We get discounted rewards
            discounted_rewards = self.get_discounted_rewards(rewards[:, i])
            # We train the critic network
            self.loss_critic[i] = self.critic.train_on_batch(observations[:, i], discounted_rewards)
        self.memory.clear()


class DealerRL(RLAgent):
    def __init__(
        self,
        N,
        obs_space_shape,
        action_space_shape,
        gamma,
        alpha,
        beta,
        temp,
        lambd,
        epsilon,
        hidden_conv_layers=[],
        hidden_dense_layers=[128],
        initializer='random_normal',
        verbose=False
    ):
        super().__init__(
            N,
            obs_space_shape,
            gamma,
            alpha,
            beta,
            temp,
            lambd,
            epsilon,
            hidden_conv_layers,
            hidden_dense_layers,
            initializer,
            verbose
        )
        self.action_space_shape = action_space_shape
        self.max_amount = int((action_space_shape[1] - 1) / 2)
        self.name = 'Dealers'
        self.build_network()

    def build_network(self):
        """
        This function is used to build the neural network of the agent
        """
        obs = tf.keras.Input(shape=self.obs_space_shape, name="dealers_obs")
        advantages = tf.keras.Input(shape=(1,), name="dealers_advantages")
        x = self.get_base_network(obs, 'dealers', 'actor')
        probability_distrib_parameters = [
            tf.keras.layers.Dense(self.action_space_shape[1], activation='softmax',
                                  kernel_initializer=self.initializer,
                                  bias_initializer=self.initializer,
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
            return -K.mean(K.minimum(ratio * advantages_with_entropy, clipped_ratio * advantages_with_entropy), keepdims=True) / self.N

        self.actor.compile(loss=actor_loss, optimizer=self.optimizer_actor, experimental_run_tf_function=False)

        x = self.get_base_network(obs, 'dealers', 'critic')
        values = tf.keras.layers.Dense(1, activation='linear',
                                       kernel_initializer=self.initializer,
                                       bias_initializer=self.initializer,
                                       name="dealers_values")(x)

        self.critic = tf.keras.Model(inputs=obs, outputs=values, name='Dealers_Critic')
        self.critic.compile(loss='mse', optimizer=self.optimizer_critic, experimental_run_tf_function=False)

        if self.verbose:
            utils.print_to_output(title='Dealers Networks')
            self.actor.summary()
            print()
            self.critic.summary()

    def get_action_simple_policy(self, observation):
        """This function takes as input an observation of one dealer
        and output an amount to trade between -self.max_amount and
        +self.max_amount FOR EACH COMPANY as an array. The total
        number of companies if self.action_space_shape[0].
        e.g. return [-2,0,2] if you want the dealer to
            • sell 2 stocks of company 0
            • do nothing for company 1
            • buy 2 stocks of company 2

        Arguments:
            observation {[np.ndarray]} -- [The dealer observation array described in the report]

        Returns:
            action {[int]} -- [an amount to trade between -self.max_amount and +self.max_amount]
        """
        print(observation)
        # sys.exit()  # Uncomment this line to stop the program just after the print
        # TO COMPLETE (replace [0]*self.action_space_shape[0])
        action = [0] * self.action_space_shape[0]
        assert(all([act <= self.max_amount and act >= -self.max_amount for act in action]))
        return action

    def get_actions(self, observations, nbs_simple_policy):
        if nbs_simple_policy:
            if nbs_simple_policy < len(observations):
                nn_output = self.policy(observations[:-nbs_simple_policy])
                if isinstance(nn_output, list):
                    probabilities = tf.stack(nn_output, axis=1).numpy()
                else:
                    probabilities = tf.expand_dims(nn_output, axis=1).numpy()
                cumsums = np.cumsum(probabilities, axis=-1)
                unif_draws = np.random.rand(*cumsums.shape)
                actions = (unif_draws < cumsums).argmax(axis=-1) - self.max_amount
                simple_policy_actions = np.array([self.get_action_simple_policy(obs) for obs in observations[-nbs_simple_policy:]], dtype=np.int32)
                return np.vstack((actions, simple_policy_actions))
            else:
                return np.array([self.get_action_simple_policy(obs) for obs in observations], dtype=np.int32)
        else:
            nn_output = self.policy(observations)
            if isinstance(nn_output, list):
                probabilities = tf.stack(nn_output, axis=1).numpy()
            else:
                probabilities = tf.expand_dims(nn_output, axis=1).numpy()
            cumsums = np.cumsum(probabilities, axis=-1)
            unif_draws = np.random.rand(*cumsums.shape)
            actions = (unif_draws < cumsums).argmax(axis=-1)
            return actions - self.max_amount

    def learn(self, epochs=1, batch_size=8):
        # We retrieve all states, actions and reward the agent got during the episode from the memory
        observations, actions, rewards = map(np.array, zip(*self.memory))
        # We rescale actions
        actions += self.max_amount
        for i in range(observations.shape[1]):
            # We process the states values with the critic network
            critic_values = np.squeeze(self.critic(observations[:, i]).numpy())
            # We one-hot encode actions
            actions_list = [tf.keras.utils.to_categorical(actions[:, i, id_], num_classes=self.action_space_shape[1]) for id_ in range(self.action_space_shape[0])]
            # We get the advantage (difference between the discounted reward and the baseline)
            advantages = self.get_advantages(critic_values, rewards[:, i])
            # We normalize advantages
            advantages = utils.normalize(advantages)
            # We train the actor network
            loss = self.actor.train_on_batch([observations[:, i], advantages], actions_list)
            if isinstance(loss, list):
                self.loss_actor[i] = loss[0]
            else:
                self.loss_actor[i] = loss
            # We get discounted rewards
            discounted_rewards = self.get_discounted_rewards(rewards[:, i])
            # We train the critic network
            self.loss_critic[i] = self.critic.train_on_batch(observations[:, i], discounted_rewards)
        self.memory.clear()
