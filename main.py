from utils import simulation
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt


def init_env(nb_companies=10, initial_nb_shares=1000, nb_dealers=50, initial_dealer_budget=10000, initial_market_maker_budget=10000, verbose=0):
    assert(initial_nb_shares % nb_dealers == 0)
    env = simulation.Market()
    env.create_companies('init.csv', nb_companies, initial_nb_shares, initial_market_maker_budget, verbose)
    env.create_dealers(nb_dealers, initial_dealer_budget, initial_nb_shares // nb_dealers, verbose)
    return env


def simulate(nb_steps, env, verbose=0, print_evolution=True):
    simulation.print_to_output(title='Simulation')
    horizon = range(nb_steps) if verbose > 0 else tqdm(range(nb_steps), total=nb_steps)
    for _ in horizon:
        actions_market_makers = env.sample_actions_market_makers()
        actions_dealers = env.sample_actions_dealers()
        env.step(actions_market_makers, actions_dealers, verbose, print_evolution)
    if print_evolution:
        plt.show()


if __name__ == "__main__":
    # Initialize the market
    env = init_env(verbose=1)
    # Run the simulation for N steps
    N = 1000
    simulate(N, env, verbose=0, print_evolution=True)
    env.reset()
