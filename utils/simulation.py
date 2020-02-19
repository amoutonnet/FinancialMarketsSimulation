# Importing libraries for calculations
import numpy as np
import pandas as pd
import random
# Importing libraries for plots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
# Importing useful libraries
import sys
from tqdm import tqdm
from multiprocessing import Pool
# Importing RL functions
from . import learning


MAX_AMOUNT = 7


def print_to_output(message=None, title=None, overoneline=True):
    """
    This function is used to simplify the printing task. You can
    print a message with a title, over one line or not for lists.
    """

    if title is not None:
        print('\n' + title.center(70, '-'))
    if message is not None:
        if isinstance(message, list):
            if overoneline:
                to_print = ''
                for i in message:
                    to_print += '%s | ' % str(i)
                print(to_print[:-3])
            else:
                for i in message:
                    print(i)
        else:
            print(message)


class Actor():
    """
    This mother class defines the basic attributes of an
    actor on the market. This actor will have an ID, a name,
    and the market he is a part of.
    """

    def __init__(self, id_, name, market):
        self.id_ = id_
        self.name = name
        self.market = market


class MarketMaker(Actor):
    """
    This class is used to represent a market maker. It inherits
    from the Actor class. The market maker operates for a company,
    by fixing the ask and the bid price. He tries to find the
    supply and demand equilibrium.
    """

    def __init__(self, id_, name, market, initial_budget, company):
        super().__init__(id_, name, market)
        self.initial_cash = initial_budget
        self.company = company      # The company he operates for
        self.company.set_market_maker(self)
        self.reset()

    def reset(self):
        self.portfolio = 0             # Its portfolio of stock for this company
        self.sales = 0                 # The number of stocks he sold last time step
        self.purchases = 0             # The number of stocks he purchased last time step
        self.position = None           # Its position on the market, represented by a Position class
        self.cash = self.initial_cash  # The cash of the market maker

    def get_state(self):
        """ This function is called to get the state of the environment seen
        by the market maker """
        if self.position is None:
            return None
        else:
            return {'position': self.position, 'sales': self.sales, 'purchases': self.purchases}

    def sample_action(self):
        state = self.get_state()
        if state is None:
            ask = 100
            spread = np.random.gamma(10)
        else:
            ask = state['position'].ask_price + np.random.normal(scale=5)
            spread = np.random.gamma(10)
        max_bid = float('inf')
        return [ask, ask - spread, max_bid]

    def make_market(self, action):
        """ This function is called at each time step and
        allow the market maker to take a position on the
        market """
        self.position = Position(self, action[0], self.portfolio, action[1], action[2])
        self.sales = 0
        self.purchases = 0

    def __str__(self):
        return "Market Maker %s" % self.id_

    __repr__ = __str__


class Dealer(Actor):
    """
    This class is used to represent a Dealer. It inherits
    from the Actor class. The dealer starts the game with
    a set of stocks, and an initial amount of cash and will try
    to maximize its profit by trading stocks.
    """

    def __init__(self, id_, name, market, initial_budget, initial_shares):
        super().__init__(id_, name, market)
        self.initial_cash = initial_budget
        self.initial_portfolio = {}
        for id_company in self.market.companies.keys():
            self.initial_portfolio[id_company] = initial_shares
        self.reset()

    def reset(self):
        self.cash = self.initial_cash                       # The cash of the dealer
        self.portfolio = self.initial_portfolio.copy()   # The dealer's portfolio of stocks

    def get_state(self):
        """ This function is called to get the state of the environment seen
        by the dealer """
        return {'portfolio': self.portfolio}

    def sample_action(self):
        # Random Action
        state = self.get_state()
        type_of_transaction = np.random.randint(3)
        if type_of_transaction == 0:
            id_company = random.choice(list(state['portfolio'].keys()))
            amount = np.random.randint(1, MAX_AMOUNT)
        elif type_of_transaction == 1:
            id_company = random.choice([k for k, v in state['portfolio'].items() if v > 0])
            if state['portfolio'][id_company] > 1:
                amount = np.random.randint(1, state['portfolio'][id_company])
            else:
                amount = 1
        else:
            id_company = None
            amount = 0
        return [type_of_transaction, id_company, amount]

    def trade(self, action, verbose):
        """ This function is called at each time step and
        allow the dealer to either buy a stock at the market
        price, sell it at the market price, or do nothing
        depending on the action he decides to take """
        type_of_transaction = action[0]
        if type_of_transaction < 2:
            company = self.market.companies[action[1]]
            amount = action[2]
            return self.get_state(), ImmediateOrder(self.market, type_of_transaction, self, company, amount).fulfill_order(verbose)

    def __str__(self):
        return ("Dealer ID%s" % self.id_).ljust(11, ' ')

    __repr__ = __str__


class Position(list):
    """
    This class inheriting from a list is used to represent a
    Position of a market maker.
    """

    def __init__(self, market_maker, ask_price, max_ask, bid_price, max_bid):
        super().__init__([market_maker, ask_price, max_ask, bid_price, max_bid])
        self.market_maker = market_maker  # The concerned market maker
        self.ask_price = ask_price      # The ask price of the current position
        self.max_ask = max_ask          # The maximum stocks the market maker is willing to sell
        self.bid_price = bid_price      # The bid price of the current position
        self.max_bid = max_bid          # The maximum stocks the market maker is willing to buy

    def __str__(self):
        return "%s is ready to buy %s stocks at %.2f and sell %d stocks at %.2f" % (self.market_maker, str(self.max_bid), self.bid_price, self.max_ask, self.ask_price)

    __repr__ = __str__


class Order():
    """ This class is used to represent an order passed by a dealer """

    def __init__(self, market, type_, contracted_by, company, amount):
        self.type_ = type_                    # The type of order, 0 is buy, 1 is sell
        self.contracted_by = contracted_by    # The dealer that contracted the order
        self.company = company                # The concerned company
        self.amount = amount                  # The amount of stocks exchanged
        self.market = market                  # The market in which the order has been passed


class ImmediateOrder(Order):
    """ This class inheriting from order is used to represent an immediate order,
    i.e. an order to be fulfilled during this time step at market price."""

    def __init__(self, market, type_, contracted_by, company, amount):
        super().__init__(market, type_, contracted_by, company, amount)

    def fulfill_order(self, verbose=False):
        """ This function is called when we want to fulfill this order """
        if verbose:
            print_to_output('\nNew order processed: %s' % str(self))
        if self.type_:
            # If it is a sell order, we process it as a sell order
            fully_fulfilled, reward = self.process_sale(self.company.market_maker.position, verbose)
        else:
            # If it is a buy order, we process it as a buy order
            fully_fulfilled, reward = self.process_purchase(self.company.market_maker.position, verbose)
        if verbose:
            if fully_fulfilled:
                print_to_output('Order was fully fullfilled')
            else:
                print_to_output('Order was not fullfilled at all due to lack of supply')
        return reward

    def process_sale(self, position, verbose):
        """ This function is called to process a sell order """
        fully_fulfilled, nb_stock_transfered, total_of_transaction = False, 0, 0
        cash_limit = position.market_maker.cash // position.bid_price
        if position.max_bid > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, position.max_bid, cash_limit)
            total_of_transaction = nb_stock_transfered * position.bid_price
            self.contracted_by.cash += total_of_transaction
            self.contracted_by.portfolio[self.company.id_] -= nb_stock_transfered
            position.market_maker.cash -= total_of_transaction
            position.market_maker.portfolio += nb_stock_transfered
            position.market_maker.purchases += 1
            position.max_bid -= nb_stock_transfered
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just sold %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company.id_, position.market_maker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        return fully_fulfilled, {'amount_processed': nb_stock_transfered, 'cash_involved': total_of_transaction}

    def process_purchase(self, position, verbose):
        """ This function is called to process a buy order """
        fully_fulfilled, nb_stock_transfered, total_of_transaction = False, 0, 0
        cash_limit = self.contracted_by.cash // position.ask_price
        if position.max_ask > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, position.max_ask, cash_limit)
            total_of_transaction = nb_stock_transfered * position.ask_price
            self.contracted_by.cash -= total_of_transaction
            self.contracted_by.portfolio[self.company.id_] += nb_stock_transfered
            position.market_maker.cash += total_of_transaction
            position.market_maker.portfolio -= nb_stock_transfered
            position.max_ask -= nb_stock_transfered
            position.market_maker.sales += 1
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just bought %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, position.market_maker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        return fully_fulfilled, {'amount_processed': nb_stock_transfered, 'cash_involved': total_of_transaction}

    def __str__(self):
        action = 'sell' if self.type_ else 'buy'
        return '%s wants to %s %d %s stocks' % (self.contracted_by, action, self.amount, self.company)

    __repr__ = __str__


class Market():
    """
    This class is used to represent a market, with a
    certain number of companies, market makers and
    dealers evolving in it
    """

    def __init__(self):
        self.market_makers = {}       # A dictionnary containing market makers
        self.dealers = {}             # A dictionnary containing dealers
        self.companies = {}           # A dictionnary containing companies
        # A few dictionnaries to track the simulation
        self.asks = {}
        self.bids = {}
        self.dealers_cash = {}
        self.market_makers_cash = {}
        self.max_steps_animation = 100
        self.animation_started = False
        self.size_of_animation = 0

    def create_companies(self, file, nb_companies, initial_nb_shares, initial_market_maker_budget, verbose):
        """ This function is used to create a company within the market """
        company_data = pd.read_csv(file)
        assert(nb_companies > 0 and nb_companies < len(company_data) and initial_nb_shares > 0 and initial_market_maker_budget > 0)
        self.market_makers_initial_cash = initial_market_maker_budget
        if verbose:
            print_to_output(title='Created Companies')
        for (_, j) in company_data[:nb_companies].iterrows():
            name = j[0]
            id_company = j[1]
            if id_company not in self.companies.keys():
                self.companies[id_company] = Company(id_company, name, self, initial_nb_shares)
                self.market_makers[id_company] = MarketMaker(id_company, 'Market Maker %s' % id_company, self, initial_market_maker_budget, self.companies[id_company])
                self.market_makers_cash[id_company] = [initial_market_maker_budget]
                self.asks[id_company] = []
                self.bids[id_company] = []
                if verbose:
                    print_to_output('%s (%s)' % (name, id_company))
        print_to_output('The %d associated Market Makers have been created' % nb_companies, 'Created Market Makers')

    def create_dealers(self, nb_dealers, initial_dealer_budget, initial_shares, verbose):
        assert(nb_dealers > 0 and initial_dealer_budget > 0 and initial_shares > 0)
        self.dealers_initial_cash = initial_dealer_budget
        for id_ in range(nb_dealers):
            self.dealers[id_] = Dealer(id_, 'Dealer ID%s' % id_, self, initial_dealer_budget, initial_shares)
            self.dealers_cash[id_] = [initial_dealer_budget]
        self.dealers_list = list(self.dealers)
        if verbose:
            print_to_output('%d dealers have been created' % nb_dealers, 'Created Dealers')

    def prepare_next_step(self):
        """ This function is used between each step to prepare placeholders
        to track the state of the market """
        for id_ in self.companies.keys():
            self.asks[id_] = self.asks[id_][-(self.max_steps_animation - 1):] + [self.market_makers[id_].position.ask_price]
            self.bids[id_] = self.bids[id_][-(self.max_steps_animation - 1):] + [self.market_makers[id_].position.bid_price]
            self.market_makers_cash[id_] = self.market_makers_cash[id_][-(self.max_steps_animation - 1):] + [self.market_makers[id_].cash]
        for id_ in self.dealers.keys():
            self.dealers_cash[id_] = self.dealers_cash[id_][-(self.max_steps_animation - 1):] + [self.dealers[id_].cash]
        self.size_of_animation = min(self.size_of_animation + 1, self.max_steps_animation)

    def get_state(self, id_company):
        """ This function is used to get the state of the market, i.e. the last
        transaction that has been processed """

    def settle_positions(self, actions, verbose):
        for action, mm in zip(actions, self.market_makers.values()):
            mm.make_market(action)
        if verbose > 1:
            print_to_output([mm.position for mm in self.market_makers.values()], 'Positions of Market Makers', overoneline=False)

    def settle_trading(self, actions, verbose):
        shuffled_dealers_list = random.sample(self.dealers_list, len(self.dealers_list))
        for id_d in shuffled_dealers_list:
            self.dealers[id_d].trade(actions[id_d], verbose > 1)

    def step(self, actions_market_makers, actions_dealers, verbose, animate=False):
        if animate and not self.animation_started:
            plt.ion()
            self.animation_fig = plt.figure(figsize=(10, 8))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=self.animation_fig)
            self.ax1 = self.animation_fig.add_subplot(spec[0, :])
            self.ax2 = self.animation_fig.add_subplot(spec[1, 0])
            self.ax3 = self.animation_fig.add_subplot(spec[1, 1])
            self.ax1.set_title('Evolution of Mean Price')
            self.ax1.set_xlabel('Time (steps)')
            self.ax1.set_ylabel('Price ($)')
            self.ax1.set_ylim([0, 200])
            self.ax2.set_title('Evolution of Dealers Cash')
            self.ax2.set_xlabel('Time (steps)')
            self.ax2.set_ylabel('Cash ($)')
            self.ax2.set_ylim([0, 20000])
            self.ax3.set_title('Evolution of Market Makers Cash')
            self.ax3.set_xlabel('Time (steps)')
            self.ax3.set_ylim([0, 20000])
            self.animation_started = True
            self.animation_fig.canvas.draw()
            # self.ax1background = self.animation_fig.canvas.copy_from_bbox(self.ax1.bbox)
            # self.ax2background = self.animation_fig.canvas.copy_from_bbox(self.ax2.bbox)
            # self.ax3background = self.animation_fig.canvas.copy_from_bbox(self.ax3.bbox)
            plt.show(block=False)
            self.polygon1 = [None for i in range(len(self.asks))]
        self.settle_positions(actions_market_makers, verbose)
        self.settle_trading(actions_dealers, verbose)
        self.prepare_next_step()
        if verbose:
            print_to_output(self, 'Last Market Positions')
        if animate:
            x = np.arange(0, self.size_of_animation + 1, 1)
            if self.size_of_animation <= 1:
                for i, k in enumerate(self.asks.keys()):
                    v1, v2 = self.asks[k], self.bids[k]
                    self.ax1.plot(x[1:], (np.array(v1) + np.array(v2)) / 2, label=k)
                    # self.polygon1[i] = self.ax1.fill_between(x=x, y1=v1, y2=v2, alpha=0.3)
                for i, k in enumerate(self.dealers_cash.keys()):
                    v = self.dealers_cash[k]
                    self.ax2.plot(x, v, label=k)
                for i, k in enumerate(self.market_makers_cash.keys()):
                    v = self.market_makers_cash[k]
                    self.ax3.plot(x, v, label=k)
                # self.ax1.legend()
                self.line1 = self.ax1.get_lines()
                self.line2 = self.ax2.get_lines()
                self.line3 = self.ax3.get_lines()
            else:
                # self.animation_fig.canvas.restore_region(self.ax1background)
                # self.animation_fig.canvas.restore_region(self.ax2background)
                # self.animation_fig.canvas.restore_region(self.ax3background)
                for i, k in enumerate(self.asks.keys()):
                    v1, v2 = self.asks[k][-1], self.bids[k][-1]
                    self.line1[i].set_data(x[1:][-99:], np.append(self.line1[i].get_ydata(), [(v1 + v2) / 2])[-99:])
                    self.ax1.set_xlim([x[1], x[-1]])
                    # self.ax1.draw_artist(self.line1[i])
                for i, k in enumerate(self.dealers_cash.keys()):
                    v = self.dealers_cash[k][-1:]
                    self.line2[i].set_data(x[-99:], np.append(self.line2[i].get_ydata(), v)[-99:])
                    self.ax2.set_xlim([x[0], x[-1]])
                    # self.ax2.draw_artist(self.line2[i])
                for i, k in enumerate(self.market_makers_cash.keys()):
                    v = self.market_makers_cash[k][-1:]
                    self.line3[i].set_data(x[-99:], np.append(self.line3[i].get_ydata(), v)[-99:])
                    self.ax3.set_xlim([x[0], x[-1]])
                #     self.ax3.draw_artist(self.line3[i])
                # self.animation_fig.canvas.blit(self.ax1.bbox)
                # self.animation_fig.canvas.blit(self.ax2.bbox)
                # self.animation_fig.canvas.blit(self.ax3.bbox)
                self.animation_fig.canvas.draw()
                self.animation_fig.canvas.flush_events()

    def sample_actions_market_makers(self):
        actions = []
        for mm in self.market_makers.values():
            actions.append(mm.sample_action())
        return actions

    def sample_actions_dealers(self):
        actions = []
        for d in self.dealers.values():
            actions.append(d.sample_action())
        return actions

    def reset(self):
        for dealer in self.dealers.values():
            dealer.reset()
        for id_company in self.companies.keys():
            self.market_makers[id_company].reset()
            self.asks[id_company] = []
            self.bids[id_company] = []
            self.dealers_cash[id_company] = [self.dealers_initial_cash]
            self.market_makers_cash[id_company] = [self.market_makers_initial_cash]

    def __str__(self):
        to_print = ''
        for id_company in self.companies.keys():
            to_print += "%s | Ask: $%.2f | Bid: $%.2f\n" % (id_company.ljust(5), self.bids[id_company][-1], self.bids[id_company][-1])
        return to_print


class Company(Actor):
    """
    This class is used to represent a Company. It inherits
    from the Actor class. The company stocks are traded at a certain
    price over the market, and each company is assigned a market
    dealer responsible for making a market happen
    """

    def __init__(self, id_, name, market, initial_nb_shares):
        super().__init__(id_, name, market)
        self.total_shares_on_market = initial_nb_shares          # The total number of shares over the market
        self.market_maker = None

    def set_market_maker(self, market_maker):
        self.market_maker = market_maker

    def __str__(self):
        return self.id_.ljust(5)

    __repr__ = __str__
