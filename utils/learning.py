import numpy as np
import random
from . import simulation
import sys


def get_reward_market_makers(observation):
    """
    observation is an array of size (4, window_size + 1)
    Rows represent the state of some caracteristics during the window_size last time steps plus the current one

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


def get_reward_dealers(observation):
    """
    observation is an array of size (3*nb_companies + 7, window_size + 1)
    Rows represent the state of some caracteristics during the window_size last time steps plus the current one.
    Here the current time steps is not fully updated as the training phase occurs when the step is not finished.
    Only the state of the market (1st 2*nb_companies rows), the portfolio value and the global wealth is updated.
    Therefore you should only consider the first window_size columns as the last state, the last column will be used
    to take a decision according to the current state of the market.

    Here are the descriptions of the rows:

    - The 1st (2*nb_companies) rows represent the state of the market. For i in 1...nb_companies, the ith row
    is the ask price of the company i fixed for the time step, and the (i+1)th row is the portfolio of the market maker handling
    company i (what was available to sell at the beginning of the time step)

    - The following nb_companies rows represent the state of the portfolio of the dealer. For i in 1...nb_companies,
    the ith row is the amount of stocks issued by company i owned by the dealer at the end of the time step.

    - The last 7 rows are defined as follows:
        • The 1st one is the cash of the dealer in dollars at the end of the time step.
        • The 2nd one is the portfolio value in dollars at the beginning of the time step.
        • The 3rd one is the global wealth of the dealer in dollars (sum of the two previous rows) at the beginning of the time step.
        • The 4th one is the index of the company traded during the time step
        • The 5th one is the action taken (0 for buy, 1 for sell, 2 for doing nothing) during the time step
        • The 6th one is the amount traded during the time step
        • The 7th one is the success of this transaction (0 for not processed transaction, 1 for partially processed and 2 for fully processed) during the time step
    """
    # TO COMPLETE
    reward = None

    return reward


def save_experience_market_makers(obs, action, reward):
    pass


def save_experience_dealers(obs, action, reward):
    pass


def get_actions_market_makers(env, epsilon=1):
    actions = [None] * len(env.market_makers)
    for id_mm, mm in env.market_makers.items():
        if np.random.rand() <= epsilon:
            actions[id_mm] = mm.sample_action()
        else:
            observation = mm.get_observation()
    return actions


def get_actions_dealers(env, epsilon=1):
    actions = [None] * len(env.dealers)
    for id_d, d in env.dealers.items():
        if np.random.rand() <= epsilon:
            actions[id_d] = d.sample_action()
        else:
            observation = d.get_observation()
    return actions


def train_dealers(env, action):
    for d in env.dealers.values():
        observation = d.get_observation()
        reward = get_reward_dealers(observation)
        save_experience_dealers(observation, action, reward)


def train_market_makers(env, action):
    for mm in env.market_makers.values():
        observation = mm.get_observation()
        reward = get_reward_market_makers(observation)
        save_experience_market_makers(observation, action, reward)
