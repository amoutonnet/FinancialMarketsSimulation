from utils import simulation
from utils import learning
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import sys


def simulate(nb_steps, env, verbose=0, animate=True, plot_final=True):
    simulation.print_to_output(title='Simulation Starting')
    horizon = range(nb_steps) if verbose > 0 else tqdm(range(nb_steps), total=nb_steps)
    if animate:
        env.init_animation()
    for i in horizon:
        actions_mm = learning.get_actions_market_makers(env)
        env.settle_positions(actions_mm, verbose)
        if i > 0:
            learning.train_dealers(env, actions_d)
        actions_d = learning.get_actions_dealers(env)
        env.settle_trading(actions_d, verbose)
        learning.train_market_makers(env, actions_mm)
        if animate:
            env.step_animation()
        env.prepare_next_step()
    if plot_final:
        env.plot_final()


if __name__ == "__main__":
    N = 2
    animation = False
    plot_final = True
    verbose = 2
    nb_companies = 10
    initial_nb_shares = 1000
    nb_dealers = 50
    initial_dealer_budget = 10000
    initial_price = 100
    window_size = 20
    spread = 5
    # Initialize the market
    env = simulation.Market(nb_companies, initial_nb_shares, nb_dealers, initial_dealer_budget, initial_price, window_size, spread, verbose=1, follow_hist_data=plot_final)
    # Run the simulation for N steps
    simulate(N, env, verbose=verbose, animate=animation, plot_final=plot_final)
    env.reset()
