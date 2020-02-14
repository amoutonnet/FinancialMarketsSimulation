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
        self.portfolio = {c: 0 for c in companies}
        self.list_of_companies = companies
        self.ask_price = {c: [] for c in companies}
        self.max_ask = {c: [] for c in companies}
        self.bid_price = {c: [] for c in companies}
        self.max_bid = {c: [] for c in companies}
        self.sell_number = 0
        self.buy_number = 0
        pass

    def make_market(self):
        for c in self.list_of_companies:
            state = self.market.get_state(c)
            if state is None:
                initial_ask = 100
                spread = np.random.gamma(1)
                initial_bid = initial_ask - spread
                self.ask_price[c] += [initial_ask]
                self.max_ask[c] += [0]
                self.bid_price[c] += [initial_bid]
                self.max_bid[c] += [np.random.randint(7)]
                self.market.take_position(self, c, initial_ask, 0, initial_bid, self.max_bid[c][-1])
            else:
                ask = state[2] + np.random.normal()
                spread = np.random.gamma(1)
                bid = ask - spread
                self.ask_price[c] += [ask]
                self.max_ask[c] += [np.random.randint(self.portfolio[c])] if self.portfolio[c] else [0]
                self.bid_price[c] += [bid]
                self.max_bid[c] += [np.random.randint(7)]
                self.market.take_position(self, c, ask, self.max_ask[c][-1], bid, self.max_bid[c][-1])

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

    def take_action(self, epsilon=1):
        if np.random.rand() <= epsilon:
            id_company = random.choice(list(self.portfolio))
            state = self.market.get_state(id_company)
            action = np.random.randint(3)
            if action == 0:
                if state is not None:
                    self.buymarket(id_company, np.random.randint(7), state[2] + np.random.gamma(1))
                else:
                    self.buymarket(id_company, np.random.randint(7), float('inf'))
            elif action == 1:
                if state is not None:
                    self.sellmarket(id_company, np.random.randint(min(self.portfolio[id_company], 7)), state[0] - np.random.gamma(1))
                else:
                    self.sellmarket(id_company, np.random.randint(min(self.portfolio[id_company], 7)), 0)
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

    def __str__(self):
        return ("Dealer ID%s" % self.id_).ljust(11, ' ')

    __repr__ = __str__


class Position(list):
    def __init__(self, marketmaker, ask_price, max_ask, bid_price, max_bid):
        super().__init__([marketmaker, ask_price, max_ask, bid_price, max_bid])
        self.marketmaker = marketmaker
        self.ask_price = ask_price
        self.max_ask = max_ask
        self.bid_price = bid_price
        self.max_bid = max_bid

    def __str__(self):
        return "%s is ready to buy %d stocks at %.2f and sell %d stocks at %.2f" % (self.marketmaker, self.max_bid, self.bid_price, self.max_ask, self.ask_price)

    __repr__ = __str__


class PositionsList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, position):
        super().append(position)
        self.descending_order = sorted(self, key=itemgetter(3), reverse=True)
        self.ascending_order = sorted(self, key=itemgetter(1))


class Order():
    def __init__(self, market, type_, contracted_by, company, amount):
        self.type_ = type_
        self.contracted_by = contracted_by
        self.company = company
        self.amount = amount
        self.market = market


class ImmediateOrder(Order):
    def __init__(self, market, type_, contracted_by, company, amount, extreme_price):
        super().__init__(market, type_, contracted_by, company, amount)
        self.extreme_price = extreme_price

    def fulfill_order(self, positions, verbose=False):
        if verbose:
            output('\nNew order: %s' % str(self))
        if self.type_:
            fully_fulfilled, transaction_happend, over_extreme = self.process_sale(positions.descending_order, verbose)
        else:
            fully_fulfilled, transaction_happend, over_extreme = self.process_purchase(positions.ascending_order, verbose)
        if verbose:
            if fully_fulfilled:
                output('Order was fully fullfilled')
            elif transaction_happend:
                if over_extreme:
                    output('Order was not fully fullfilled due to volatile market')
                else:
                    output('Order was not fully fullfilled due to lack of supply')
            else:
                if over_extreme:
                    output('Order was not fullfilled at all due to volatile market')
                else:
                    output('Order was not fullfilled at all due to lack of supply')

    def process_sale(self, positions, verbose):
        fully_fulfilled = False
        transaction_happend = False
        over_extreme = False
        for j in range(len(positions)):
            if self.extreme_price > positions[j].bid_price:
                over_extreme = True
                break
            if positions[j].max_bid > 0 and self.contracted_by.cash >= positions[j].bid_price * positions[j].max_bid:
                transaction_happend = True
                nb_stock_transfered = min(self.amount, positions[j].max_bid)
                total_of_transaction = nb_stock_transfered * positions[j].bid_price
                self.contracted_by.cash += total_of_transaction
                self.contracted_by.portfolio[self.company] -= nb_stock_transfered
                positions[j].marketmaker.cash -= total_of_transaction
                positions[j].marketmaker.portfolio[self.company] += nb_stock_transfered
                positions[j].max_bid -= nb_stock_transfered
                self.amount -= nb_stock_transfered
                if verbose:
                    output('New transaction: %s just sold %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, positions[j].marketmaker, total_of_transaction))
                if self.amount == 0:
                    fully_fulfilled = True
                    break
        return fully_fulfilled, transaction_happend, over_extreme

    def process_purchase(self, positions, verbose):
        fully_fulfilled = False
        transaction_happend = False
        over_extreme = False
        for j in range(len(positions)):
            if self.extreme_price < positions[j].ask_price:
                over_extreme = True
                break
            if positions[j].max_ask > 0 and self.contracted_by.cash >= positions[j].ask_price * positions[j].max_ask:
                transaction_happend = True
                nb_stock_transfered = min(self.amount, positions[j].max_ask)
                total_of_transaction = nb_stock_transfered * positions[j].ask_price
                self.contracted_by.cash -= total_of_transaction
                self.contracted_by.portfolio[self.company] += nb_stock_transfered
                positions[j].marketmaker.cash += total_of_transaction
                positions[j].marketmaker.portfolio[self.company] -= nb_stock_transfered
                positions[j].max_ask -= nb_stock_transfered
                self.amount -= nb_stock_transfered
                if verbose:
                    output('New transaction: %s just bought %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, positions[j].marketmaker, total_of_transaction))
                if self.amount == 0:
                    fully_fulfilled = True
                    break
        return fully_fulfilled, transaction_happend, over_extreme

    def __str__(self):
        action = 'sell' if self.type_ else 'buy'
        to_print = '%s wants to %s %d %s stocks' % (self.contracted_by, action, self.amount, self.company)
        if self.type_ and self.extreme_price > 0:
            to_print += ' for %.2f minimum' % self.extreme_price
        elif not self.type_ and self.extreme_price < float('inf'):
            to_print += ' for %.2f maximum' % self.extreme_price
        return to_print

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
            self.positions[id_] = [PositionsList()]
        else:
            output("This company's symbol already exists")

    def reset_positions(self):
        for id_ in self.companies.keys():
            self.positions[id_] = [PositionsList()]

    def create_dealer(self):
        id_ = len(self.dealers)
        self.dealers[id_] = Dealer(id_, 'Dealer ID%s' % id_, self, self.initial_budget_per_dealer, self.initial_portfolio)

    def create_marketmaker(self, companies):
        id_ = len(self.marketmakers)
        self.marketmakers[id_] = MarketMaker(id_, 'Market Maker ID%s' % id_, self, self.initial_budget_per_marketmaker, companies)

    def create_immediate_order(self, type_, contracted_by, id_company, amount, extreme):
        self.immediate_orders += [ImmediateOrder(self, type_, contracted_by, id_company, amount, extreme)]

    def take_position(self, market_maker, id_company, ask_price, max_ask, bid_price, max_bid):
        self.positions[id_company][-1].append(Position(market_maker, ask_price, max_ask, bid_price, max_bid))

    def get_state(self, id_company):
        if self.companies[id_company].is_market:
            return (self.companies[id_company].bid_price_market[-1], self.companies[id_company].max_bid_market[-1],
                    self.companies[id_company].ask_price_market[-1], self.companies[id_company].max_ask_market[-1])
        else:
            return None

    def time_step(self, verbose):
        for mm in self.marketmakers.values():
            mm.make_market()
        for d in self.dealers.values():
            d.take_action()
        if verbose:
            for i, j in self.positions.items():
                output(j[-1], i)
            output(self.immediate_orders, 'Immediate Orders')
        random.shuffle(self.immediate_orders)
        for order in self.immediate_orders:
            order.fulfill_order(self.positions[order.company][-1], True)
        if verbose:
            print(self)
        self.immediate_orders = []
        self.reset_positions()
        self.current_time_step += 1

    def simulate(self, nb_steps=20, verbose=False):
        for _ in range(nb_steps):
            self.time_step(verbose)

    def __str__(self):
        to_print = '\n' + ('State of the market at time %d' % self.current_time_step).center(70, '-')
        is_market_somewhere = False
        for company in self.companies.values():
            if company.is_market:
                is_market_somewhere = True
                to_print += "\n%s | Ask: $%.2f x %d | Bid: $%.2f x %d" % (
                    company, company.ask_price_market[-1], company.max_ask_market[-1], company.bid_price_market[-1], company.max_bid_market[-1])
        if not is_market_somewhere:
            to_print += "\nThere is no market for the moment, no transaction are being processed."
        return to_print


class Company(Actor):
    def __init__(self, id_, name, market, initial_nb_shares):
        super().__init__(id_, name, market)
        self.total_shares_on_market = initial_nb_shares
        self.is_market = False
        self.bid_price_market = []
        self.max_bid_market = []
        self.ask_price_market = []
        self.max_ask_market = []

    def __str__(self):
        return self.id_.ljust(5)

    __repr__ = __str__


if __name__ == "__main__":
    market = Market()
    market.init()
    market.simulate(1, False)
