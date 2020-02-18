# Importing libraries for calculations
import numpy as np
import pandas as pd
import random
# Importing libraries for plots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Importing useful libraries
import sys
from tqdm import tqdm


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
        self.cash = initial_budget  # The cash of the market maker
        self.company = company      # The company he operates for
        self.portfolio = 0          # Its portfolio of stock for this company
        self.sells = 0              # The number of stocks he sold last time step
        self.purchases = 0          # The number of stocks he purchased last time step
        self.position = None        # Its position on the market, represented by a Position class

    def get_state(self):
        """ This function is called to get the state of the environment seen
        by the market maker """
        if self.position is None:
            return None
        else:
            return [self.position, self.sells, self.purchases]

    def policy(self, epsilon=1):
        """ This function is called to get the action to take for the time step """
        state = self.get_state()
        if np.random.rand() <= epsilon:
            # Random Action
            if state is None:
                ask = 100
                spread = np.random.gamma(10)
            else:
                ask = state[0].ask_price + np.random.normal(scale=5)
                spread = np.random.gamma(10)
        else:
            # Thoughtful action
            ask = 0
            spread = 0
        max_bid = 1000
        return [ask, ask - spread, max_bid]

    def make_market(self, action):
        """ This function is called at each time step and
        allow the market maker to take a position on the
        market """
        self.position = Position(self, action[0], self.portfolio, action[1], action[2])
        self.sells = 0
        self.purchases = 0

    def __str__(self):
        return ("Market Maker %s" % self.id_)

    __repr__ = __str__


class Dealer(Actor):
    """
    This class is used to represent a Dealer. It inherits
    from the Actor class. The dealer starts the game with
    a set of stocks, and an initial amount of cash and will try
    to maximize its profit by trading stocks.
    """

    def __init__(self, id_, name, market, initial_budget, initial_portfolio):
        super().__init__(id_, name, market)
        self.cash = initial_budget           # The cash of the dealer
        self.portfolio = initial_portfolio   # The dealer's portfolio of stocks

    def get_state(self):
        """ This function is called to get the state of the environment seen
        by the dealer """
        return None

    def policy(self, epsilon=1):
        """ This function is called to get the action to take for the time step """
        state = self.get_state()
        if np.random.rand() <= epsilon:
            # Random Action
            type_of_transaction = np.random.randint(3)
            if type_of_transaction == 0:
                id_company = random.choice(list(self.portfolio.keys()))
                amount = np.random.randint(1, MAX_AMOUNT)
                extreme_price = float('inf')
            elif type_of_transaction == 1:
                id_company = random.choice([k for k, v in self.portfolio.items() if v > 0])
                if self.portfolio[id_company] > 1:
                    amount = np.random.randint(1, self.portfolio[id_company])
                else:
                    amount = 1
                extreme_price = 0
            else:
                id_company = None
                amount = 0
                extreme_price = 0
        else:
            # Thoughtful Action
            id_company = None
            type_of_transaction = 0
            amount = 0
        return [type_of_transaction, id_company, amount, extreme_price]

    def trade(self, action):
        """ This function is called at each time step and
        allow the dealer to either buy a stock at the market
        price, sell it at the market price, or do nothing
        depending on the action he decides to take """
        type_of_transaction = action[0]
        if type_of_transaction < 2:
            id_company = action[1]
            amount = action[2]
            extreme_price = action[3]
            self.market.create_immediate_order(type_of_transaction, self, id_company, amount, extreme_price)

    def __str__(self):
        return ("Dealer ID%s" % self.id_).ljust(11, ' ')

    __repr__ = __str__


class Position(list):
    """
    This class inheriting from a list is used to represent a
    Position of a market maker.
    """

    def __init__(self, marketmaker, ask_price, max_ask, bid_price, max_bid):
        super().__init__([marketmaker, ask_price, max_ask, bid_price, max_bid])
        self.marketmaker = marketmaker  # The concerned market maker
        self.ask_price = ask_price      # The ask price of the current position
        self.max_ask = max_ask          # The maximum stocks the market maker is willing to sell
        self.bid_price = bid_price      # The bid price of the current position
        self.max_bid = max_bid          # The maximum stocks the market maker is willing to buy

    def __str__(self):
        return "%s is ready to buy %d stocks at %.2f and sell %d stocks at %.2f" % (self.marketmaker, self.max_bid, self.bid_price, self.max_ask, self.ask_price)

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

    def __init__(self, market, type_, contracted_by, company, amount, extreme_price):
        super().__init__(market, type_, contracted_by, company, amount)
        self.extreme_price = extreme_price  # The price below (resp above) which this sell (resp buy) order is cancelled

    def fulfill_order(self, position, verbose=False):
        """ This function is called when we want to fulfill this order """
        if verbose:
            print_to_output('\nNew order processed: %s' % str(self))
        if self.type_:
            # If it is a sell order, we process it as a sell order
            fully_fulfilled, over_extreme = self.process_sale(position, verbose)
        else:
            # If it is a buy order, we process it as a buy order
            fully_fulfilled, over_extreme = self.process_purchase(position, verbose)
        if verbose:
            if fully_fulfilled:
                print_to_output('Order was fully fullfilled')
            else:
                if over_extreme:
                    print_to_output('Order was not fullfilled at all due to volatile market')
                else:
                    print_to_output('Order was not fullfilled at all due to lack of supply')

    def process_sale(self, position, verbose):
        """ This function is called to process a sell order """
        fully_fulfilled = False
        over_extreme = False
        if self.extreme_price > position.bid_price:
            over_extreme = True
        cash_limit = position.marketmaker.cash // position.bid_price
        if position.max_bid > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, position.max_bid, cash_limit)
            total_of_transaction = nb_stock_transfered * position.bid_price
            self.contracted_by.cash += total_of_transaction
            self.contracted_by.portfolio[self.company] -= nb_stock_transfered
            position.marketmaker.cash -= total_of_transaction
            position.marketmaker.portfolio += nb_stock_transfered
            position.marketmaker.purchases += 1
            position.max_bid -= nb_stock_transfered
            self.market.companies[self.company].bid_price_market[-1] = position.bid_price
            self.market.companies[self.company].max_bid_market[-1] = nb_stock_transfered
            self.market.companies[self.company].is_market_bid = True
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just sold %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, position.marketmaker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        return fully_fulfilled, over_extreme

    def process_purchase(self, position, verbose):
        """ This function is called to process a buy order """
        fully_fulfilled = False
        over_extreme = False
        if self.extreme_price < position.ask_price:
            over_extreme = True
        cash_limit = self.contracted_by.cash // position.ask_price
        if position.max_ask > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, position.max_ask, cash_limit)
            total_of_transaction = nb_stock_transfered * position.ask_price
            self.contracted_by.cash -= total_of_transaction
            self.contracted_by.portfolio[self.company] += nb_stock_transfered
            position.marketmaker.cash += total_of_transaction
            position.marketmaker.portfolio -= nb_stock_transfered
            position.max_ask -= nb_stock_transfered
            position.marketmaker.sells += 1
            self.market.companies[self.company].ask_price_market[-1] = position.ask_price
            self.market.companies[self.company].max_ask_market[-1] = nb_stock_transfered
            self.market.companies[self.company].is_market_ask = True
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just bought %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, position.marketmaker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        return fully_fulfilled, over_extreme

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
    """
    This class is used to represent a market, with a
    certain number of companies, market makers and
    dealers evolving in it
    """

    def __init__(self, nb_dealer=100, nb_companies=10, initial_nb_shares=1000, initial_budget_per_dealer=10000, initial_budget_per_marketmaker=10000):
        assert(initial_nb_shares % nb_dealer == 0)
        self.current_time_step = 0                                            # Initializing the time step of the simulation
        self.nb_dealer = nb_dealer                                            # The number of dealers
        self.initial_nb_shares = initial_nb_shares                            # The number of initial shares per dealer
        self.initial_budget_per_dealer = initial_budget_per_dealer            # The initial budget for each dealer
        self.initial_budget_per_marketmaker = initial_budget_per_marketmaker  # The initial budget for each market maker
        self.nb_companies = nb_companies                                      # The number of companies
        self.companies = {}                                                   # A dictionnary containing companies
        self.dealers = {}                                                     # A dictionnary containing dealers
        self.immediate_orders = []                                            # A dictionnary containing immediate orders

        # A few dictionnaries to track the simulation
        self.asks = {}
        self.bids = {}
        self.dealers_cash = {}
        self.marketmakers_cash = {}

    def init_simulation(self):
        """ This function is called to initialize the simulation (companies, dealers...) """
        self.initial_portfolio = {}
        companies_info = pd.read_csv('init.csv')[:self.nb_companies]
        for (_, j) in companies_info.iterrows():
            self.create_company(j[0], j[1])
            self.initial_portfolio[j[1]] = int(self.initial_nb_shares / self.nb_dealer)
        print_to_output(list(self.companies.keys()), 'Created Companies')
        print_to_output('There was %d market makers created, one for each company.' % self.nb_companies, 'Created Market Makers')
        for _ in range(self.nb_dealer):
            self.create_dealer()
        print_to_output('There was %d dealers created.' % self.nb_dealer, 'Created Dealers')

    def create_company(self, name, id_):
        """ This function is used to create a company within the market """
        if id_ not in self.companies.keys():
            self.companies[id_] = Company(id_, name, self, self.initial_nb_shares)
            self.asks[id_] = []
            self.bids[id_] = []
        else:
            print_to_output("This company's symbol already exists")

    def prepare_next_step(self):
        """ This function is used between each step to prepare placeholders
        to track the state of the market """
        for id_ in self.companies.keys():
            self.asks[id_] += [self.companies[id_].marketmaker.position.ask_price]
            self.bids[id_] += [self.companies[id_].marketmaker.position.bid_price]
            self.companies[id_].is_market_ask = False
            self.companies[id_].is_market_bid = False
            self.companies[id_].bid_price_market += [None]
            self.companies[id_].max_bid_market += [None]
            self.companies[id_].ask_price_market += [None]
            self.companies[id_].max_ask_market += [None]
            self.marketmakers_cash[id_] += [self.companies[id_].marketmaker.cash]
        for id_ in self.dealers.keys():
            self.dealers_cash[id_] += [self.dealers[id_].cash]

    def create_dealer(self):
        """ This function is used to create a dealer within the market """
        id_ = len(self.dealers)
        self.dealers[id_] = Dealer(id_, 'Dealer ID%s' % id_, self, self.initial_budget_per_dealer, self.initial_portfolio.copy())
        self.dealers_cash[id_] = [self.initial_budget_per_dealer]

    def create_immediate_order(self, type_, contracted_by, id_company, amount, extreme):
        """ This function is used to create an immediate order within the market """
        self.immediate_orders += [ImmediateOrder(self, type_, contracted_by, id_company, amount, extreme)]

    def get_state(self, id_company):
        """ This function is used to get the state of the market, i.e. the last
        transaction that has been processed """
        state = []
        if self.companies[id_company].is_market_ask:
            state += [self.companies[id_company].ask_price_market, self.companies[id_company].max_ask_market]
        else:
            state += [None, None]
        if self.companies[id_company].is_market_bid:
            state += [self.companies[id_company].bid_price_market, self.companies[id_company].max_bid_market]
        else:
            state += [None, None]
        return state

    def time_step(self, verbose):
        """ This function is used to simulate a time step """
        # First each market maker takes a position
        for company in self.companies.values():
            company.marketmaker.make_market(company.marketmaker.policy())
        if verbose > 1:
            print_to_output([i.marketmaker.position for i in self.companies.values()], 'Positions of Market Makers', overoneline=False)
        # Then each dealer takes an action which creates an order
        for d in self.dealers.values():
            d.trade(d.policy())
        if verbose > 1:
            print_to_output(title='Orders')
        # Then we fulfill each order randomly
        random.shuffle(self.immediate_orders)
        for order in self.immediate_orders:
            order.fulfill_order(self.companies[order.company].marketmaker.position, verbose > 1)
        if verbose:
            print(self)

        # And prepare next step
        self.immediate_orders = []
        self.prepare_next_step()
        self.current_time_step += 1

    def simulate(self, nb_steps=20, verbose=0):
        """ This function is used to simulate a certain number of time steps """
        self.init_simulation()
        horizon = range(nb_steps) if verbose > 0 else tqdm(range(nb_steps), total=nb_steps)
        for _ in horizon:
            self.time_step(verbose)
        self.plot_final_state(nb_steps)

    def plot_final_state(self, nb_steps):
        """ This function is used to plot the information we tracked during the simulation """
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
        ax1 = fig.add_subplot(spec[0, :])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])
        x = np.arange(0, nb_steps, 1)
        for (k, v1), v2 in zip(self.asks.items(), self.bids.values()):
            ax1.plot(x, (np.array(v1) + np.array(v2)) / 2, label=k)
            ax1.fill_between(x=x, y1=v1, y2=v2, alpha=0.3)
        ax1.legend()
        ax1.set_title('Evolution of Mean Price')
        ax1.set_xlabel('Time (steps)')
        ax1.set_ylabel('Price ($)')
        for k, v in self.dealers_cash.items():
            ax2.plot(v, label=k)
        ax2.set_title('Evolution of Dealers Cash')
        ax2.set_xlabel('Time (steps)')
        ax2.set_ylabel('Cash ($)')
        for k, v in self.marketmakers_cash.items():
            ax3.plot(v, label=k)
        ax3.set_title('Evolution of Market Makers Cash')
        ax3.set_xlabel('Time (steps)')
        ax3.set_ylabel('Cash ($)')
        plt.show()

    def __str__(self):
        to_print = '\n' + ('Last transactions at time %d' % self.current_time_step).center(70, '-')
        for company in self.companies.values():
            info_company = '\n%s ' % company
            if company.is_market_ask:
                info_company += "| Ask: $%.2f x %d " % (company.ask_price_market[-1], company.max_ask_market[-1])
            else:
                info_company += '|' + '-' * 17
            if company.is_market_bid:
                info_company += '| Bid: $%.2f x %d ' % (company.bid_price_market[-1], company.max_bid_market[-1])
            else:
                info_company += '|' + '-' * 17
            to_print += info_company
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
        self.is_market_ask = False                               # A boolean telling if there is dealers are buying
        self.is_market_bid = False                               # A boolean telling if there is dealers are selling
        self.bid_price_market = [None]                           # An array tracking the price of selling transactions from dealers
        self.max_bid_market = [None]                             # An array tracking the amount of selling transactions from dealers
        self.ask_price_market = [None]                           # An array tracking the price of buying transactions from dealers
        self.max_ask_market = [None]                             # An array tracking the amount of buying transactions from dealers

        # The market maker assigned to the company
        self.marketmaker = MarketMaker(id_, 'Market Maker %s' % id_, self.market, self.market.initial_budget_per_marketmaker, self)
        self.market.marketmakers_cash[self.id_] = [self.marketmaker.cash]

    def __str__(self):
        return self.id_.ljust(5)

    __repr__ = __str__


if __name__ == "__main__":
    # Initialize the market
    market = Market()
    # Run the simulation for N steps
    N = 100
    market.simulate(N, verbose=0)
