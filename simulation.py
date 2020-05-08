from utils import utils
from utils import environmentv2
from utils import learning
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import sys
import time
import tensorflow.keras.initializers as init

SEED = 20


tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Simulation():
    def __init__(
        self,
        Nm,
        nb_market_makers_using_simple_policy,
        Nd,
        nb_dealers_using_simple_policy,
        T,
        p0,
        m0,
        c0,
        A0,
        W,
        S,
        L,
        M,
        mm_parameters,
        d_parameters,
    ):
        assert(
            nb_market_makers_using_simple_policy <= Nm
            and nb_market_makers_using_simple_policy >= 0
        )
        assert(
            nb_dealers_using_simple_policy <= Nd
            and nb_dealers_using_simple_policy >= 0
        )
        self.nb_market_makers_using_simple_policy = nb_market_makers_using_simple_policy  # List of simple policy market makers
        self.nb_dealers_using_simple_policy = nb_dealers_using_simple_policy  # List of simple policy market makers
        self.env = environmentv2.Market(Nm, Nd, T, p0, m0, c0, A0, S, W, L, M)
        self.mm_agent = learning.MarketMakerRL(self, Nm, (2 * Nm + 2 * Nd, W), (utils.DELTA, M), **mm_parameters)
        self.d_agent = learning.DealerRL(self, Nd, (2 * Nm + 3 * Nm + 1, W), (Nm, 2 * L + 1), **d_parameters)

    def simulate_random(self, plot_final, print_states, print_rewards, verbose=0):
        utils.print_to_output(title='Random Simulated Trading Day')
        self.env.reset()
        observations_mm, _ = self.env.get_all_mm()
        while(self.env.step < self.env.T + self.env.W):
            if print_states:
                print(observations_mm)
            self.env.fix_market(self.env.get_mm_random_actions())
            if self.env.step == self.env.W:
                observations_d, _ = self.env.get_all_d()
            else:
                observations_d, rewards_d = self.env.get_all_d()
                if print_rewards:
                    print(rewards_d)
            if print_states:
                print(observations_d)
            self.env.pass_orders(self.env.get_d_random_actions())
            self.env.process_orders()
            self.env.prepare_next()
            observations_mm, rewards_mm = self.env.get_all_mm()
            if print_rewards:
                print(rewards_mm)
        self.env.fix_market(self.env.get_mm_random_actions())
        _, rewards_d = self.env.get_all_d()
        if print_rewards:
            print(rewards_d)
        self.env.show_env('Random Simulated Trading Day')

    def train(self, max_episodes=100, process_average_over=10, test_every=50, test_on=5):
        utils.print_to_output(title='Training Starting')
        ep = 0
        mm_training_score = np.empty((self.env.Nm, max_episodes))
        mm_training_rolling_average = np.empty((self.env.Nm, max_episodes))
        d_training_score = np.empty((self.env.Nd, max_episodes))
        d_training_rolling_average = np.empty((self.env.Nd, max_episodes))
        while ep < max_episodes:
            mm_episode_reward = np.zeros((self.env.Nm,))
            d_episode_reward = np.zeros((self.env.Nd,))
            self.env.reset()
            observations_mm, _ = self.env.get_all_mm()
            while(self.env.step < self.env.T + self.env.W):
                actions_mm = self.mm_agent.get_actions(observations_mm, self.nb_market_makers_using_simple_policy, self.env.step - self.env.W)
                self.env.fix_market(actions_mm)
                if self.env.step == self.env.W:
                    observations_d, _ = self.env.get_all_d()
                else:
                    next_observations_d, rewards_d = self.env.get_all_d()
                    d_episode_reward += rewards_d
                    self.d_agent.memory.append((observations_d, actions_d, rewards_d))
                    observations_d = next_observations_d
                actions_d = self.d_agent.get_actions(observations_d, self.nb_dealers_using_simple_policy, self.env.step - self.env.W)
                self.env.pass_orders(actions_d)
                self.env.process_orders()
                self.env.prepare_next()
                next_observations_mm, reward_mm = self.env.get_all_mm()
                mm_episode_reward += reward_mm
                self.mm_agent.memory.append((observations_mm, actions_mm, reward_mm))
                observations_mm = next_observations_mm
            actions_mm = self.mm_agent.get_actions(observations_mm, self.nb_market_makers_using_simple_policy, self.env.step - self.env.W)
            self.env.fix_market(actions_mm)
            _, rewards_d = self.env.get_all_d()
            d_episode_reward += rewards_d
            self.d_agent.memory.append((observations_d, actions_d, rewards_d))
            self.mm_agent.learn(self.nb_market_makers_using_simple_policy)
            self.d_agent.learn(self.nb_dealers_using_simple_policy)
            mm_training_score[:, ep] = mm_episode_reward
            mm_training_rolling_average[:, ep] = np.mean(mm_training_score[:, max(0, ep - process_average_over):ep + 1], axis=1)
            d_training_score[:, ep] = d_episode_reward
            d_training_rolling_average[:, ep] = np.mean(d_training_score[:, max(0, ep - process_average_over):ep + 1], axis=1)
            print('Episode {:5d}/{:5d}'.format(ep + 1, max_episodes))
            print(self.mm_agent.name)
            self.mm_agent.print_verbose(mm_training_score[:, ep], mm_training_rolling_average[:, ep])
            print(self.d_agent.name)
            self.d_agent.print_verbose(d_training_score[:, ep], d_training_rolling_average[:, ep])
            if (ep + 1) % test_every <= test_on:  # and ep > test_on:
                self.env.show_env()
            ep += 1
        _, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(mm_training_score[0], 'b', linewidth=1, label='Score')
        ax[0].plot(mm_training_rolling_average[0], 'orange', linewidth=1, label='Rolling Average')
        for i in range(1, len(mm_training_score)):
            ax[0].plot(mm_training_score[i], 'b', linewidth=1)
            ax[0].plot(mm_training_rolling_average[i], 'orange', linewidth=1)
        ax[0].set_title('Evolution of market makers score during the Training')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Score')
        ax[0].legend()
        ax[1].plot(d_training_score[0], 'b', linewidth=1, label='Score')
        ax[1].plot(d_training_rolling_average[0], 'orange', linewidth=1, label='Rolling Average')
        for i in range(1, len(d_training_score)):
            ax[1].plot(d_training_score[i], 'b', linewidth=1)
            ax[1].plot(d_training_rolling_average[i], 'orange', linewidth=1)
        ax[1].set_title('Evolution of dealers score during the Training')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Score')
        ax[1].legend()
        plt.show()


if __name__ == "__main__":
    T = 1000
    nb_companies = 1
    nb_market_makers_using_simple_policy = 1  # This number needs to be between 0 and nb_companies
    initial_nb_shares_per_market_maker = 1000
    initial_price = 100
    nb_dealers = 1
    nb_dealers_using_simple_policy = 1  # This number needs to be between 0 and nb_dealers
    initial_nb_shares_per_dealer_per_company = 100
    initial_dealer_budget = 10000
    window_size = 5
    spread = 5
    L = 10
    M = 1e8
    verbose = False
    mm_parameters = {
        'gamma': 0.99,
        'alpha': 1e-3,
        'beta': 1e-3,
        'temp': 0.001,
        'lambd': 0.5,
        'epsilon': 0.2,
        'hidden_conv_layers': [(64, 4), (32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'initializer': init.RandomNormal(),
        'verbose': verbose,
        'simple_policy_dict': {
            'option': 1,
            'sin_mean': 100,
            'sin_amplitude': 10,
            'sin_period': 20,
            'random_walk_step': 2,
            'brown_scale': 5
        }
    }
    d_parameters = {
        'gamma': 0.99,
        'alpha': 1e-5,
        'beta': 1e-5,
        'temp': 1,
        'lambd': 0.5,
        'epsilon': 0.1,
        'hidden_conv_layers': [(64, 4), (32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'initializer': init.he_normal(),
        'verbose': verbose,
        'simple_policy_dict': {
            'option': 2,
            'sell_baseline': 105,
            'buy_baseline': 95,
            'simple_buy_amount': 5,
            'simple_sell_amount': 5,
        }
    }

    # Initialize the market
    sim = Simulation(
        nb_companies,   # The number of companies to include in the simulation
        nb_market_makers_using_simple_policy,   # The number of market makers whose actions will be dicted by a simple policy and not by RL
        nb_dealers,   # The number of dealers to inlcude in the simulation
        nb_dealers_using_simple_policy,   # The number of dealers whose actions will be dicted by a simple policy and not by RL
        T,   # The length of the trading day
        initial_nb_shares_per_dealer_per_company,   # The initial number of shares per dealer per company
        initial_nb_shares_per_market_maker,   # The initial number of shares per market maker
        initial_dealer_budget,   # The initial amount of cash of dealers
        initial_price,   # The initial ask price
        window_size,   # The length of the observable window size
        spread,   # The constant spread during the simulation
        L,
        M,
        mm_parameters,   # The parameters for the market maker RL agent
        d_parameters   # The parameters for the dealer RL agent
    )
    # Run the simulation for T steps
    # sim.simulate_random(plot_final=True, print_states=False, print_rewards=False)
    sim.train(max_episodes=1000, process_average_over=50, test_every=2, test_on=1)
