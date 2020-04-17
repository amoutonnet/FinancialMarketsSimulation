from utils import environment
from utils import learning
from utils import utils
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import deque
import sys
import time
import copy
import numpy as np


class Simulation():
    def __init__(self, Nm, Nd, T, p0, c0, A0, W, S, mm_parameters, d_parameters):
        self.Nm = Nm  # Number of market makers
        self.T = T  # Length of trading day
        self.p0 = p0  # Initial stocks per dealer per company
        self.Nd = Nd  # Number of dealers
        self.c0 = c0  # Initial cash per dealer
        self.A0 = A0  # Initial ask price for every company
        self.W = W  # Window length
        self.S = S  # Spread
        self.summarize_initial_state()
        self.env = environment.Market(Nm, Nd, T, p0, c0, A0, W, S)
        mm_obs_space_shape = self.env.get_market_makers_observations_shape()
        mm_actions_limits = self.env.get_marker_makers_actions_limits()
        self.mm_agent = learning.MarketMakerRL(Nm, mm_obs_space_shape, mm_actions_limits, **mm_parameters)
        d_obs_space_shape = self.env.get_dealers_observations_shape()
        d_actions_space_shape = self.env.get_dealers_actions_shape()
        self.d_agent = learning.DealerRL(Nd, d_obs_space_shape, d_actions_space_shape, **d_parameters)

    def summarize_initial_state(self):
        utils.print_to_output(message=["%s:%s" % (str(key), str(val)) for key, val in self.__dict__.items()], title='Simulation Initialization')

    def simulate_random(self, plot_final, print_states, print_rewards, verbose=0, length=None):
        utils.print_to_output(title='Random Simulated Trading Day')
        if length is None:
            length = self.T
        self.env.reset()
        observations_mm, _ = self.env.get_response_market_makers()
        for i in tqdm(range(length), total=length):
            if print_states:
                print(observations_mm)
            actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
            self.env.settle_positions(actions_mm, verbose)
            if i == 0:
                observations_d, _ = self.env.get_response_dealers()
            else:
                observations_d, rewards_d = self.env.get_response_dealers()
                if print_rewards:
                    print(rewards_d)
            if print_states:
                print(observations_d)
            actions_d = list(map(lambda d: d.sample_action(), self.env.dealers.values()))
            self.env.settle_trading(actions_d, verbose)
            self.env.prepare_next_step()
            observations_mm, rewards_mm = self.env.get_response_market_makers()
            if print_rewards:
                print(rewards_mm)
        actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
        self.env.settle_positions(actions_mm, verbose)
        _, rewards_d = self.env.get_response_dealers()
        if print_rewards:
            print(rewards_d)
        self.env.show_env('Random Simulated Trading Day')

    def train(self, max_episodes=100, process_average_over=10, test_every=50, test_on=5):
        utils.print_to_output(title='Training Starting')
        ep = 0
        mm_training_score = np.empty((self.Nm, max_episodes))
        mm_training_rolling_average = np.empty((self.Nm, max_episodes))
        d_training_score = np.empty((self.Nd, max_episodes))
        d_training_rolling_average = np.empty((self.Nd, max_episodes))
        while ep < max_episodes:
            mm_episode_reward = np.zeros((self.Nm,))
            d_episode_reward = np.zeros((self.Nd))
            self.env.reset()
            observations_mm, _ = self.env.get_response_market_makers()
            for i in range(self.T):
                actions_mm = self.mm_agent.get_actions(observations_mm)
                self.env.settle_positions(actions_mm)
                if i == 0:
                    observations_d, _ = self.env.get_response_dealers()
                else:
                    next_observations_d, rewards_d = self.env.get_response_dealers()
                    d_episode_reward += rewards_d
                    self.d_agent.memory.append((observations_d, actions_d, rewards_d))
                    observations_d = next_observations_d
                actions_d = self.d_agent.get_actions(observations_d)
                self.env.settle_trading(actions_d)
                self.env.prepare_next_step()
                next_observations_mm, reward_mm = self.env.get_response_market_makers()
                mm_episode_reward += reward_mm
                self.mm_agent.memory.append((observations_mm, actions_mm, reward_mm))
                observations_mm = next_observations_mm
            actions_mm = self.mm_agent.get_actions(observations_mm)
            self.env.settle_positions(actions_mm)
            _, rewards_d = self.env.get_response_dealers()
            d_episode_reward += rewards_d
            self.d_agent.memory.append((observations_d, actions_d, rewards_d))
            if (ep + 1) % test_every <= test_on:
                self.env.show_env()
            else:
                self.env.close_env()
            self.mm_agent.learn()
            self.d_agent.learn()
            mm_training_score[:, ep] = mm_episode_reward
            mm_training_rolling_average[:, ep] = np.mean(mm_training_score[:, max(0, ep - process_average_over):ep + 1], axis=1)
            d_training_score[:, ep] = d_episode_reward
            d_training_rolling_average[:, ep] = np.mean(d_training_score[:, max(0, ep - process_average_over):ep + 1], axis=1)
            print('Episode {:5d}/{:5d}'.format(ep + 1, max_episodes))
            self.mm_agent.print_verbose(ep + 1, max_episodes, np.mean(mm_training_score[:, ep]), np.mean(mm_training_rolling_average[:, ep]))
            self.d_agent.print_verbose(ep + 1, max_episodes, np.mean(d_training_score[:, ep]), np.mean(d_training_rolling_average[:, ep]))
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
    T = 100
    nb_companies = 5
    initial_nb_shares = 100
    nb_dealers = 20
    initial_dealer_budget = 10000
    initial_price = 10
    window_size = 20
    spread = 5
    mm_parameters = {
        'gamma': 0.99,
        'alpha': 1e-3,
        'beta': 1e-3,
        'temp': 1,
        'lambd': 0.5,
        'epsilon': 0.2,
        'hidden_conv_layers': [(32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'verbose': True
    }
    d_parameters = {
        'gamma': 0.99,
        'alpha': 1e-3,
        'beta': 1e-3,
        'temp': 0.01,
        'lambd': 0.5,
        'epsilon': 0.2,
        'hidden_conv_layers': [(32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'verbose': True
    }

    # Initialize the market
    sim = Simulation(nb_companies, nb_dealers, T, initial_nb_shares, initial_dealer_budget, initial_price, window_size, spread, mm_parameters, d_parameters)
    # Run the simulation for T steps
    # sim.simulate_random(plot_final=True, print_states=False, print_rewards=True)
    sim.train(max_episodes=100, process_average_over=10, test_every=50, test_on=5)
