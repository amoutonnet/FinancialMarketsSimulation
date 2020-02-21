from utils import simulation
from utils import learning
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys


def init_env(nb_companies=10, initial_nb_shares=1000, nb_dealers=50, initial_dealer_budget=10000, initial_market_maker_budget=10000, verbose=0):
    assert(initial_nb_shares % nb_dealers == 0)
    env = simulation.Market()
    initial_obs_mm = env.create_companies('init.csv', nb_companies, initial_nb_shares, initial_market_maker_budget, verbose)
    initial_obs_d = env.create_dealers(nb_dealers, initial_dealer_budget, initial_nb_shares // nb_dealers, verbose)
    return env, initial_obs_mm, initial_obs_d


def simulate(nb_steps, env, initial_obs_mm, initial_obs_d, verbose=0, animate=True, plot_final=True):
    simulation.print_to_output(title='Simulation')
    horizon = range(nb_steps) if verbose > 0 else tqdm(range(nb_steps), total=nb_steps)
    last_obs_mm, last_obs_d = initial_obs_mm, initial_obs_d
    for _ in horizon:
        actions_mm = env.sample_actions_market_makers()
        # actions_mm = learning.get_actions_market_makers()
        actions_d = env.sample_actions_dealers()
        # actions_d = learning.get_actions_dealers()
        next_obs_mm, next_obs_d = env.step(actions_mm, actions_d, verbose, animate)
        reward_mm = learning.get_reward_market_makers(last_obs_mm, next_obs_mm)
        reward_d = learning.get_reward_dealers(last_obs_d, next_obs_d)
        learning.save_experience_market_makers(last_obs_mm, actions_mm, next_obs_mm, reward_mm)
        learning.save_experience_dealers(last_obs_d, actions_d, next_obs_d, reward_d)
    if plot_final:
        env.plot_final()


if __name__ == "__main__":
    # Initialize the market
    env, initial_obs_mm, initial_obs_d = init_env(verbose=1)
    # Run the simulation for N steps
    N = 1000
    simulate(N, env, initial_obs_mm, initial_obs_d, verbose=0, animate=False, plot_final=True)
    env.reset()
