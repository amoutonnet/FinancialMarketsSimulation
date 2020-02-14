import numpy as np
import pandas as pd
import sys
import bisect
import random
from operator import itemgetter


def output(message, title=None):
    if title is not None:
        print('\n' + title.center(70, '-'))
    if isinstance(message, list):
        for i in message:
            print(i)
    else:
        print(message)


class Actor():
    def __init__(self, id_, name, market):
        self.id_ = id_
        self.name = name
        self.market = market


class MarketMaker(Actor):
    def __init__(self, id_, name, market, initial_budget, companies):
        super().__init__(id_, name, market)
        self.cash = initial_budget
        self.list_of_companies = companies
        self.ask_price = {c: [] for c in companies}
        self.bid_price = {c: [] for c in companies}
        self.sell_number = 0
        self.buy_number = 0
        pass

    def make_market(self, state=None):
        if state is None:
            for c in self.list_of_companies:
                initial_ask = np.random.rand() * 900 + 100
                spread = np.random.gamma(1)
                initial_bid = initial_ask - spread
                self.ask_price[c] += [initial_ask]
                self.bid_price[c] += [initial_bid]
                self.market.take_position(self, c, initial_ask, initial_bid)
        else:
            pass

    def __str__(self):
        return ("Market Maker ID%s" % self.id_).ljust(16, ' ')

    __repr__ = __str__


class Dealer(Actor):
    def __init__(self, id_, name, market, initial_budget, initial_portfolio):
        super().__init__(id_, name, market)
        self.cash = initial_budget
        self.portfolio = initial_portfolio
        self.shorts = {}
        self.account_value = initial_budget

    def take_action(self, state, epsilon=1):
        if np.random.rand() <= epsilon:
            id_company = random.choice(list(self.portfolio))
            action = np.random.randint(5)
            if action == 0:
                self.buymarket(id_company, np.random.randint(7), state[id_company].ask_price_market[-1] + np.random.gamma(1))
            elif action == 1:
                self.buylimit(id_company, np.random.randint(7), state[id_company].ask_price_market[-1] - np.random.gamma(1))
            elif action == 2:
                self.sellmarket(id_company, np.random.randint(min(self.portfolio[id_company], 7)), state[id_company].bid_price_market[-1] - np.random.gamma(1))
            elif action == 3:
                self.selllimit(id_company, np.random.randint(min(self.portfolio[id_company], 7)), state[id_company].bid_price_market[-1] + np.random.gamma(1))
            else:
                pass
        else:
            pass

    def buymarket(self, id_company, amount, extreme_price):
        if amount > 0:
            self.market.create_immediate_order(0, self, id_company, amount, extreme_price)

    def sellmarket(self, id_company, amount, extreme_price):
        if amount > 0:
            self.market.create_immediate_order(1, self, id_company, amount, extreme_price)

    def buylimit(self, id_company, amount, price):
        if amount > 0:
            self.market.create_limit_order(0, self, id_company, amount, price)

    def selllimit(self, id_company, amount, price):
        if amount > 0:
            self.market.create_limit_order(1, self, id_company, amount, price)

    def __str__(self):
        return ("Dealer ID%s" % self.id_).ljust(11, ' ')

    __repr__ = __str__


class Position(list):
    def __new__(cls, marketmaker, ask_price, max_ask, bid_price, max_bid):
        return super().__new__(cls, [marketmaker, ask_price, max_ask, bid_price, max_bid])

    def __init__(self, marketmaker, ask_price, max_ask, bid_price, max_bid):
        self.marketmaker = marketmaker
        self.ask_price = ask_price
        self.max_ask = max_ask
        self.bid_price = bid_price
        self.max_bid = max_bid


class Positions(list):
    def __init__(self, positions):
        self.positions_descending = sorted(positions, key=itemgetter(3), reverse=True)
        self.positions_ascending = sorted(positions, key=itemgetter(1))


class Order():
    def __init__(self, type_, contracted_by, company, amount):
        self.type_ = type_
        self.contracted_by = contracted_by
        self.company = company
        self.amount = amount


class ImmediateOrder(Order):
    def __init__(self, type_, contracted_by, company, amount, extreme_price):
        super().__init__(type_, contracted_by, company, amount)
        self.extreme_price = extreme_price

    def __str__(self):
        action = 'sell' if self.type_ else 'buy'
        extreme = 'minimum' if self.type_ else 'maximum'
        return '%s wants to %s %d %s stocks for %.2f %s' % (self.contracted_by, action.ljust(4, ' '), self.amount, self.company.ljust(5, ' '), self.extreme_price, extreme)

    __repr__ = __str__


class LimitOrder(Order):
    def __init__(self, type_, contracted_by, company, amount, price):
        super().__init__(type_, contracted_by, company, amount)
        self.trigger_price = price

    def __str__(self):
        action = 'sell' if self.type_ else 'buy'
        limit = 'over' if self.type_ else 'below'
        return '%s wants to %s %d %s stocks as soon as price go %s %.2f' % (self.contracted_by, action.ljust(4, ' '), self.amount, self.company.ljust(5, ' '), limit, self.trigger_price)

    __repr__ = __str__


class Market():
    def __init__(self, nb_dealer=50, nb_companies=10, initial_nb_shares=1000, initial_budget_per_dealer=10000,
                 nb_marketmaker_per_company=4, nb_companies_per_marketmarker=5, initial_budget_per_marketmaker=10000):
        assert(initial_nb_shares % nb_dealer == 0)
        assert(nb_companies % nb_companies_per_marketmarker == 0)
        self.current_time_step = 0
        self.nb_dealer = nb_dealer
        self.initial_nb_shares = initial_nb_shares
        self.initial_budget_per_dealer = initial_budget_per_dealer
        self.initial_budget_per_marketmaker = initial_budget_per_marketmaker
        self.nb_marketmaker_per_company = nb_marketmaker_per_company
        self.nb_companies = nb_companies
        self.nb_companies_per_marketmarker = nb_companies_per_marketmarker

        self.companies = {}
        self.dealers = {}
        self.marketmakers = {}
        self.positions = {}
        self.initial_portfolio = {}
        self.immediate_orders = []
        self.limit_orders = []

    def init(self):
        self.initial_portfolio = {}
        companies_info = pd.read_csv('init.csv')[:self.nb_companies]
        list_of_companies = []
        for (_, j) in companies_info.iterrows():
            self.create_company(j[0], j[1])
            list_of_companies += [j[1]]
            self.initial_portfolio[j[1]] = int(self.initial_nb_shares / self.nb_dealer)
        output(list_of_companies, 'Created Companies')
        for _ in range(self.nb_dealer):
            self.create_dealer()
        output('There was %d dealers created.' % self.nb_dealer, 'Created Dealers')
        total_market_makers = 0
        for _ in range(self.nb_marketmaker_per_company):
            random.shuffle(list_of_companies)
            for i in range(self.nb_companies // self.nb_companies_per_marketmarker):
                self.create_marketmaker(list_of_companies[i * self.nb_companies_per_marketmarker:(i + 1) * self.nb_companies_per_marketmarker])
                total_market_makers += 1
        output('There was %d market makers created.' % total_market_makers, 'Created Market Makers')

    def create_company(self, name, id_):
        if id_ not in self.companies.keys():
            self.companies[id_] = Company(id_, name, self, self.initial_nb_shares)
            self.positions[id_] = [[]]
        else:
            output("This company's symbol already exists")

    def create_dealer(self):
        id_ = len(self.dealers)
        self.dealers[id_] = Dealer(id_, 'Dealer ID%s' % id_, self, self.initial_budget_per_dealer, self.initial_portfolio)

    def create_marketmaker(self, companies):
        id_ = len(self.marketmakers)
        self.marketmakers[id_] = MarketMaker(id_, 'Market Maker ID%s' % id_, self, self.initial_budget_per_marketmaker, companies)

    def create_immediate_order(self, type_, contracted_by, id_company, amount, extreme_price):
        self.immediate_orders += [ImmediateOrder(type_, contracted_by, id_company, amount, extreme_price)]

    def create_limit_order(self, type_, contracted_by, id_company, amount, price):
        self.limit_orders += [LimitOrder(type_, contracted_by, id_company, amount, price)]

    def take_position(self, market_maker, id_company, ask_price, bid_price):
        self.positions[id_company][-1] += [(market_maker, ask_price, bid_price)]
        if self.companies[id_company].bid_price_market[-1] < bid_price:
            self.companies[id_company].bid_price_market[-1] = bid_price
        if self.companies[id_company].ask_price_market[-1] > ask_price:
            self.companies[id_company].ask_price_market[-1] = bid_price

    def time_step(self, verbose):
        for mm in self.marketmakers.values():
            mm.make_market()
        for d in self.dealers.values():
            d.take_action(self.companies)
        if verbose:
            for i, j in self.positions.items():
                output(j[-1], i)
            output(self.immediate_orders, 'Immediate Orders')
        random.shuffle(self.limit_orders)
        for order in self.limit_orders:
            if order.type_:
                # If the order is a sell order
                interesting_position = sorted(self.positions[order.company][-1], key=itemgetter(2), reverse=True)[0]
                if interesting_position[2] > order.trigger_price:
                    order.process(interesting_position[0])
            else:
                # If the order is a buy order
                interesting_position = sorted(self.positions[order.company][-1], key=itemgetter(1))[0]
                if interesting_position[1] < order.trigger_price:
                    order.process(interesting_position[0])
        for order in self.immediate_orders:
            if order.type_:
                # If the order is a sell order
                interesting_position = sorted(self.positions[order.company][-1], key=itemgetter(2), reverse=True)[0]
                if interesting_position[2] > order.extreme_price:
                    order.process(interesting_position[0])
            else:
                # If the order is a buy order
                interesting_position = sorted(self.positions[order.company][-1], key=itemgetter(1))[0]
                if interesting_position[1] < order.extreme_price:
                    order.process(interesting_position[0])
        self.current_time_step += 1

    def simulate(self, nb_steps=20, verbose=False):
        for _ in range(nb_steps):
            self.time_step(verbose)
            print(self)

    def __str__(self):
        to_print = '\n' + ('State of the market at time %d' % self.current_time_step).center(70, '-')
        for company in self.companies.values():
            to_print += "\n%s | Ask: $%.2f x %d | Bid: $%.2f x %d" % (
                company, company.ask_price_market[-1], company.max_ask_market[-1], company.bid_price_market[-1], company.max_bid_market[-1])
        return to_print


class Company(Actor):
    def __init__(self, id_, name, market, initial_nb_shares):
        super().__init__(id_, name, market)
        self.total_shares_on_market = initial_nb_shares
        self.bid_price_market = [0]
        self.max_bid_market = [0]
        self.ask_price_market = [float('inf')]
        self.max_ask_market = [0]

    def __str__(self):
        return self.id_.ljust(5)

    __repr__ = __str__


if __name__ == "__main__":
    market = Market()
    market.init()
    market.simulate(1)
