from utils import environment
from utils import learning
from utils import utils
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
import time
import copy


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
        self.mm_agent = learning.MarketMakerRL(mm_obs_space_shape, mm_actions_limits, **mm_parameters)
        d_obs_space_shape = self.env.get_dealers_observations_shape()
        d_actions_space_shape = self.env.get_dealers_actions_shape()
        self.d_agent = learning.DealerRL(d_obs_space_shape, d_actions_space_shape, **d_parameters)

    def summarize_initial_state(self):
        utils.print_to_output(message=["%s:%s" % (str(key), str(val)) for key, val in self.__dict__.items()], title='Simulation Initialization')

    def simulate_random(self, plot_final, animate, print_states, print_rewards, verbose=0, length=None):
        utils.print_to_output(title='Random Simulated Trading Day')
        if length is None:
            length = self.T
        if animate:
            self.env.init_animation()
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
        if plot_final:
            self.env.plot_final('Random Simulated Trading Day')

    def train(self, max_episodes=100):
        ep = 0
        while ep < max_episodes:
            self.env.reset()
            observations_mm, _ = self.env.get_response_market_makers()
            for i in range(self.T):
                actions_mm = self.mm_agent.get_actions(observations_mm)
                self.env.settle_positions(actions_mm)
                if i == 0:
                    observations_d, _ = self.env.get_response_dealers()
                else:
                    next_observations_d, rewards_d = self.env.get_response_dealers()
                    self.d_agent.memory.append((observations_d, actions_d, rewards_d, next_observations_d))
                    observations_d = next_observations_d
                actions_d = self.d_agent.get_actions(observations_d)
                self.env.settle_trading(actions_d)
                self.env.prepare_next_step()
                next_observations_mm, reward_mm = self.env.get_response_market_makers()
                self.mm_agent.memory.append((observations_mm, actions_mm, reward_mm, next_observations_mm))
                observations_mm = next_observations_mm
            actions_mm = self.mm_agent.get_actions(observations_mm)
            self.env.settle_positions(actions_mm)
            next_observations_mm, rewards_d = self.env.get_response_dealers()
            self.d_agent.memory.append((observations_d, actions_d, rewards_d, next_observations_d))
            ep += 1
            self.mm_agent.learn()
            self.d_agent.learn()
            sys.exit()


if __name__ == "__main__":
    T = 10
    nb_companies = 2
    initial_nb_shares = 100
    nb_dealers = 3
    initial_dealer_budget = 10000
    initial_price = 100
    window_size = 5
    spread = 5
    mm_parameters = {
        'gamma': 0.99,
        'alpha': 1e-3,
        'beta': 1e-3,
        'lambd': 1e-3,
        'epsilon': 0.2,
        'hidden_conv_layers': [(32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'verbose': False
    }
    d_parameters = {
        'gamma': 0.99,
        'alpha': 1e-3,
        'beta': 1e-3,
        'lambd': 1e-3,
        'epsilon': 0.2,
        'hidden_conv_layers': [(32, 3), (16, 3)],
        'hidden_dense_layers': [128, 64, 32],
        'verbose': False
    }

    # Initialize the market
    sim = Simulation(nb_companies, nb_dealers, T, initial_nb_shares, initial_dealer_budget, initial_price, window_size, spread, mm_parameters, d_parameters)
    # Run the simulation for T steps
    # sim.simulate_random(plot_final=True, animate=False, print_states=False, print_rewards=True, length=10)
    sim.train()
