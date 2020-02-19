import numpy as np
import random

MAX_AMOUNT = 7


def policy_market_maker(state, epsilon):
    ask = 0
    spread = 1
    max_bid = float('inf')
    return [ask, ask - spread, max_bid]


def policy_dealer(state, epsilon):
    # Thoughtful Action
    id_company = None
    type_of_transaction = 0
    amount = 0
    return [type_of_transaction, id_company, amount]
