from utils import environment
from utils import learning
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys
import time
import copy


class Simulation():
    def __init__(self, Nm, Nd, T, p0, c0, A0, W, S):
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

    def summarize_initial_state(self):
        environment.print_to_output(message=["%s:%s" % (str(key), str(val)) for key, val in self.__dict__.items()], title='Simulation Initialization')

    def simulate_random(self, plot_final, animate, print_states, print_rewards, verbose=0, length=None):
        environment.print_to_output(title='Random Simulated Trading Day')
        if length is None:
            length = self.T
        if animate:
            self.env.init_animation()
        self.env.reset()
        observations_mm = self.env.get_observations_market_makers()
        for i in tqdm(range(length), total=length):
            actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
            self.env.settle_positions(actions_mm, verbose)
            observations_d = self.env.get_observations_dealers()
            if print_states:
                print(observations_mm)
                print(observations_d)
            if print_rewards:
                if i > 0:
                    print(self.env.get_rewards_dealers(observations_d))
            actions_d = list(map(lambda d: d.sample_action(), self.env.dealers.values()))
            self.env.settle_trading(actions_d, verbose)
            self.env.prepare_next_step()
            observations_mm = self.env.get_observations_market_makers()
            if print_rewards:
                print(self.env.get_rewards_market_makers(observations_mm))
        if print_states:
            print(observations_mm)
            print(observations_d)
        actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
        self.env.settle_positions(actions_mm, verbose)
        observations_d = self.env.get_observations_dealers()
        if print_rewards:
            print(self.env.get_rewards_dealers(observations_d))
        if plot_final:
            self.env.plot_final('Random Simulated Trading Day')

    def simulate_intelligent(self, plot_final, animate):
        horizon = tqdm(range(self.T), total=self.T)
        if animate:
            self.env.init_animation()
        self.env.reset()
        observations_mm = self.env.get_observations_market_makers()
        for i in horizon:
            actions_mm = learning.get_actions_market_makers(observations_mm)
            self.env.settle_positions(actions_mm)
            if i > 0:
                rewards_d = self.env.get_rewards_dealers(observations_d)
            observations_d = self.env.get_observations_dealers()
            actions_d = learning.get_actions_dealers(observations_d)
            self.env.settle_trading(actions_d)
            self.env.prepare_next_step()
            observations_mm = self.env.get_observations_market_makers()
            reward_mm = self.env.get_rewards_market_makers(observations_mm)
        observations_d = self.env.get_observations_dealers()
        actions_mm = learning.get_actions_market_makers(observations_mm)
        self.env.settle_positions(actions_mm)
        rewards_d = self.env.get_rewards_dealers(observations_d)
        if plot_final:
            self.env.plot_final()

    def train(self, episodes=100):
        for n in tqdm(range(episodes), total=episodes):
            self.env.reset()
            observations_mm = self.env.get_observations_market_makers()
            for i in range(2):
                print(observations_mm)
                # actions_mm = learning.get_actions_market_makers(observations_mm)
                actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
                self.env.settle_positions(actions_mm)
                observations_d = self.env.get_observations_dealers()
                print(observations_d)
                if i > 0:
                    rewards_d = self.env.get_rewards_dealers(observations_d)
                actions_d = list(map(lambda d: d.sample_action(), self.env.dealers.values()))
                # actions_d = learning.get_actions_dealers(observations_d
                self.env.settle_trading(actions_d)
                self.env.prepare_next_step()
                observations_mm = self.env.get_observations_market_makers()
                reward_mm = self.env.get_rewards_market_makers(observations_mm)
            # actions_mm = learning.get_actions_market_makers(observations_mm)
            actions_mm = list(map(lambda mm: mm.sample_action(), self.env.market_makers.values()))
            self.env.settle_positions(actions_mm)
            observations_d = self.env.get_observations_dealers()
            rewards_d = self.env.get_rewards_dealers(observations_d)


if __name__ == "__main__":
    T = 1000
    nb_companies = 2
    initial_nb_shares = 100
    nb_dealers = 2
    initial_dealer_budget = 10000
    initial_price = 100
    window_size = 5
    spread = 5
    # Initialize the market
    sim = Simulation(nb_companies, nb_dealers, T, initial_nb_shares, initial_dealer_budget, initial_price, window_size, spread)
    # Run the simulation for T steps
    sim.simulate_random(plot_final=True, animate=False, print_states=False, print_rewards=True, length=10)
