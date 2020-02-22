import numpy as np
import random
from . import simulation
import sys


def get_reward_market_makers(last_obs, new_obs):
    """
    obs is a dictionnary, whose keys correspond to market makers (e.g GOOG for market maker handling Google stocks)
    and whose values corresponds to observations. Observations are dictionnaries with the following structure:
    - portfolio (int) : amount of stocks owned by the market maker (don't forget that one market maker handle only one company)
    - cash (float) : the cash owned by the market maker
    - last_sales (int) : the amount of sales (market maker sells to dealer) done during the time step
    - last_purchases (int) : the amount of purchases (market maker buys to dealer) done during the time step
    - ask_price (float) : the ask price asked by the market maker during the time step
    - max_ask (int) : the amount of stocks the market maker was ready to sell during the time step
    - bid_price (float) : the bid price asked by the market maker during the time step
    - max_bid (int) : the amount of stocks the market maker was ready to buy during the time step
    """

    # TO COMPLETE

    pass


def get_reward_dealers(last_obs, new_obs):
    """
    obs is a dictionnary, whose keys correspond to dealers (e.g 2 for dealer no. 2)
    and whose values corresponds to observations. Observations are dictionnaries with the following structure:
    - MSFT, GOOG, AMZN... (int) : the amount of stocks owned by the dealer for the given company
    - cash (float) : the cash owned by the dealer
    - cannot_process (boolean) : True if the dealer decided to sold without having any stocks, or buy without having any cash, False else
    - fully_fulfilled (boolean) : True if the dealer managed to buy (resp sell) the total amount he wanted to buy (resp sell), no less
    - last_transaction_amount (int) : The number of stocks sold (or bought) during the time step
    - last_transaction_cost (float) : The amount of cash he earned (or spend) by selling (or buying) stocks during the time step
    - type_of_action (int) : 0 for BUY, 1 for SELL, 2 for DRAW
    """

    # TO COMPLETE

    pass


def save_experience_market_makers(last_obs, reward, new_obs, action):
    pass


def save_experience_dealers(last_obs, reward, new_obs, action):
    pass


def get_actions_market_makers():
    pass


def get_actions_market_dealers():
    pass
