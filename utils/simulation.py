# Importing libraries for calculations
import numpy as np
import pandas as pd
import random
# Importing libraries for plots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')

# For Yida
# matplotlib.use('Qt5Agg')

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


class Agent():
    """
    This mother class defines the basic attributes of an
    Agent on the market. This Agent will have an ID, a name,
    and the market he is a part of.
    """

    def __init__(self, id_, name, market):
        self.id_ = id_
        self.name = name
        self.market = market


class MarketMakerAgent(Agent):
    """
    This class is used to represent a market maker type of agent. It inherits
    from the Agent class. The market maker operates for a company,
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
        self.portfolio = 0                                       # Its portfolio of stock for this company
        self.last_sales = 0                                           # The number of stocks he sold last time step
        self.last_purchases = 0                                       # The number of stocks he purchased last time step
        self.position = Position(self, 100, 0, 100, 0)           # Its position on the market, represented by a Position class
        self.cash = self.initial_cash                            # The cash of the market maker

    def sample_action(self):
        ask = self.position.ask_price + np.random.normal(scale=5)
        spread = np.random.gamma(10)
        return [ask, ask - spread]

    def make_market(self, action):
        """ This function is called at each time step and
        allow the market maker to take a position on the
        market """
        self.position = Position(self, action[0], self.portfolio, action[1], self.cash//action[1])

    def get_observation(self):
        to_return = {'portfolio': self.portfolio, 'cash': self.cash, 'last_sales': self.last_sales, 'last_purchases': self.last_purchases}
        self.last_sales = 0
        self.last_purchases = 0
        return {**to_return, **{k:v for k,v in self.position.__dict__.items() if k!='market_maker'}}

    def __str__(self):
        return "Market Maker %s" % self.id_

    __repr__ = __str__


class DealerAgent(Agent):
    """
    This class is used to represent a Dealer type of agent. It inherits
    from the Agent class. The dealer starts the game with
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

    def sample_action(self):
        # Random Action
        type_of_transaction = np.random.randint(3)
        if type_of_transaction == 0:
            id_company = random.choice(list(self.portfolio.keys()))
            amount = np.random.randint(1, MAX_AMOUNT)
        elif type_of_transaction == 1:
            id_company = random.choice(list(self.portfolio.keys()))
            if self.portfolio[id_company] > 1:
                amount = np.random.randint(1, self.portfolio[id_company])
            else:
                amount = self.portfolio[id_company]
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
        response = None
        if type_of_transaction < 2:
            company = self.market.companies[action[1]]
            amount = action[2]
            response = ImmediateOrder(self.market, type_of_transaction, self, company, amount).fulfill_order(verbose)
        return self.get_observation(response, type_of_transaction)

    def get_observation(self, last_transaction_amount=None, type_of_transaction=2):
        if last_transaction_amount is None:
            return {**self.portfolio, 'cash': self.cash,'cannot_process': False, 'fully_fulfilled': False, 'last_transaction_amount': 0, 'last_transaction_cost': 0, 'type_of_action': 2}
        else:
            return {**self.portfolio, 'cash': self.cash, **last_transaction_amount, 'type_of_action': type_of_transaction}

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
            response = self.process_sale(self.company.market_maker.position, verbose)
        else:
            # If it is a buy order, we process it as a buy order
            response = self.process_purchase(self.company.market_maker.position, verbose)
        if verbose:
            if response['fully_fulfilled']:
                print_to_output('Order was fully fullfilled')
            elif response['cannot_process']:
                print_to_output('Order was not fullfilled because the dealer action is absurd')
            else:
                print_to_output('Order was not fullfilled at all due to lack of supply')
        return response

    def process_sale(self, position, verbose):
        """ This function is called to process a sell order """
        fully_fulfilled, cannot_process, nb_stock_transfered, total_of_transaction = False, False, 0, 0
        if position.max_bid > 0 and self.amount>0:
            nb_stock_transfered = min(self.amount, position.max_bid)
            total_of_transaction = nb_stock_transfered * position.bid_price
            self.contracted_by.cash += total_of_transaction
            self.contracted_by.portfolio[self.company.id_] -= nb_stock_transfered
            position.market_maker.cash -= total_of_transaction
            position.market_maker.portfolio += nb_stock_transfered
            position.market_maker.last_purchases += 1
            position.max_bid -= nb_stock_transfered
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just sold %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company.id_, position.market_maker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        elif self.amount==0:
            cannot_process = True
        else:
            pass
        return {'cannot_process': cannot_process, 'fully_fulfilled': fully_fulfilled, 'last_transaction_amount': nb_stock_transfered, 'last_transaction_cost': total_of_transaction}

    def process_purchase(self, position, verbose):
        """ This function is called to process a buy order """
        fully_fulfilled, cannot_process, nb_stock_transfered, total_of_transaction = False, False, 0, 0
        cash_limit = self.contracted_by.cash // position.ask_price
        if position.max_ask > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, position.max_ask, cash_limit)
            total_of_transaction = nb_stock_transfered * position.ask_price
            self.contracted_by.cash -= total_of_transaction
            self.contracted_by.portfolio[self.company.id_] += nb_stock_transfered
            position.market_maker.cash += total_of_transaction
            position.market_maker.portfolio -= nb_stock_transfered
            position.max_ask -= nb_stock_transfered
            position.market_maker.last_sales += 1
            self.amount -= nb_stock_transfered
            if verbose:
                print_to_output('Details: %s just bought %d %s stocks to %s for %.2f' % (self.contracted_by, nb_stock_transfered, self.company, position.market_maker, total_of_transaction))
            if self.amount == 0:
                fully_fulfilled = True
        elif cash_limit==0:
            cannot_process = True
        return {'cannot_process': cannot_process, 'fully_fulfilled': fully_fulfilled, 'last_transaction_amount': nb_stock_transfered, 'last_transaction_cost': total_of_transaction}

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
        self.market_makers_portfolio = {}
        self.max_steps_animation = 100
        self.max_steps = 1
        self.animation_started = False
        self.size_of_animation = 0

    def create_companies(self, file, nb_companies, initial_nb_shares, initial_market_maker_budget, verbose):
        """ This function is used to create a company within the market """
        company_data = pd.read_csv(file)
        assert(nb_companies > 0 and nb_companies < len(company_data) and initial_nb_shares > 0 and initial_market_maker_budget > 0)
        self.market_makers_initial_cash = initial_market_maker_budget
        if verbose:
            print_to_output(title='Created Companies')
        initial_observations = {}
        for (_, j) in company_data[:nb_companies].iterrows():
            name = j[0]
            id_company = j[1]
            if id_company not in self.companies.keys():
                self.companies[id_company] = Company(id_company, name, self, initial_nb_shares)
                self.market_makers[id_company] = MarketMakerAgent(id_company, 'Market Maker %s' % id_company, self, initial_market_maker_budget, self.companies[id_company])
                self.market_makers_cash[id_company] = [initial_market_maker_budget]
                self.market_makers_portfolio[id_company] = [0]
                self.asks[id_company] = [100]
                self.bids[id_company] = [100]
                initial_observations[id_company] = self.market_makers[id_company].get_observation()
                if verbose:
                    print_to_output('%s (%s)' % (name, id_company))
        if verbose:
            print_to_output('The %d associated Market Makers have been created' % nb_companies, 'Created Market Makers')
        return initial_observations

    def create_dealers(self, nb_dealers, initial_dealer_budget, initial_shares, verbose):
        assert(nb_dealers > 0 and initial_dealer_budget > 0 and initial_shares > 0)
        self.dealers_initial_cash = initial_dealer_budget
        initial_observations = {}
        for id_ in range(nb_dealers):
            self.dealers[id_] = DealerAgent(id_, 'Dealer ID%s' % id_, self, initial_dealer_budget, initial_shares)
            self.dealers_cash[id_] = [initial_dealer_budget]
            initial_observations[id_] = self.dealers[id_].get_observation()
        if verbose:
            print_to_output('%d dealers have been created' % nb_dealers, 'Created Dealers')
        return initial_observations

    def prepare_next_step(self):
        """ This function is used between each step to prepare placeholders
        to track the state of the market """
        for id_ in self.companies.keys():
            self.asks[id_] += [self.market_makers[id_].position.ask_price]
            self.bids[id_] += [self.market_makers[id_].position.bid_price]
            self.market_makers_cash[id_] += [self.market_makers[id_].cash]
            self.market_makers_portfolio[id_] += [self.market_makers[id_].portfolio]
        for id_ in self.dealers.keys():
            self.dealers_cash[id_] += [self.dealers[id_].cash]
        self.max_steps +=1
        self.size_of_animation = min(self.size_of_animation + 1, self.max_steps_animation)

    def settle_positions(self, actions, verbose):
        try:
            for id_mm in actions.keys():
                self.market_makers[id_mm].make_market(actions[id_mm])
        except KeyError:
            print_to_output('\n\nError: every market maker should have an action to do, your action dictionnary is incomplete')
            sys.exit()
        if verbose > 1:
            print_to_output([mm.position for mm in self.market_makers.values()], 'Positions of Market Makers', overoneline=False)

    def settle_trading(self, actions, verbose):
        observations = {}
        shuffled_actions = random.sample(list(actions.keys()), len(actions))
        try:
            for id_d in shuffled_actions:
                observations[id_d] = self.dealers[id_d].trade(actions[id_d], verbose > 1)
        except KeyError:
            print_to_output('\n\nError: every dealer should have an action to do, your action dictionnary is incomplete')
            sys.exit()
        return observations

    def get_companies(self):
        return list(self.companies.keys())

    def get_market_makers_observations(self):
        observations = {}
        for id_mm in self.market_makers.keys():
            observations[id_mm] = self.market_makers[id_mm].get_observation()
        return observations

    def step(self, actions_market_makers, actions_dealers, verbose, animate=False):
        if animate and not self.animation_started:
            self.init_animation()
        self.settle_positions(actions_market_makers, verbose)
        obs_d = self.settle_trading(actions_dealers, verbose)
        obs_mm = self.get_market_makers_observations()
        self.prepare_next_step()
        if verbose:
            print_to_output(self, 'Last Market Positions')
        if animate:
            self.step_animation()
        return obs_mm, obs_d

    def sample_actions_market_makers(self):
        actions = {}
        for id_mm in self.market_makers.keys():
            actions[id_mm] = self.market_makers[id_mm].sample_action()
        return actions

    def sample_actions_dealers(self):
        actions = {}
        for id_d in self.dealers.keys():
            actions[id_d] = self.dealers[id_d].sample_action()
        return actions

    def reset(self):
        for dealer in self.dealers.values():
            dealer.reset()
        for id_company in self.companies.keys():
            self.market_makers[id_company].reset()
            self.asks[id_company] = [100]
            self.bids[id_company] = [100]
            self.dealers_cash[id_company] = [self.dealers_initial_cash]
            self.market_makers_cash[id_company] = [self.market_makers_initial_cash]
            self.market_makers_portfolio[id_company] = [0]

    def init_animation(self):
        self.animation_fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=self.animation_fig)
        self.ax1 = self.animation_fig.add_subplot(spec[0, :])
        self.ax2 = self.animation_fig.add_subplot(spec[1, 0])
        self.ax3 = self.animation_fig.add_subplot(spec[2, 0])
        self.ax4 = self.animation_fig.add_subplot(spec[1, 1])
        self.ax1.set_title('Evolution of Prices', y=1.20)
        self.ax1.set_xlabel('Time (steps)')
        self.ax1.set_ylabel('Price ($)')
        self.ax1.grid()
        self.ax2.set_title('Evolution of Market Makers Cash')
        self.ax2.set_ylabel('Cash ($)')
        self.ax2.grid()
        self.ax3.set_title('Evolution of Dealers Cash')
        self.ax3.set_xlabel('Time (steps)')
        self.ax3.set_ylabel('Cash ($)')
        self.ax3.grid()
        self.ax4.set_title('Evolution of Market Makers Portfolio')
        self.ax3.set_xlabel('Time (steps)')
        self.ax4.set_ylabel('Amount')
        self.ax4.grid()
        self.animation_started = True
        self.animation_fig.canvas.draw()
        plt.show(block=False)

    def step_animation(self):
        x = np.arange(0, self.size_of_animation + 1, 1)
        if self.size_of_animation <= 1:
            for k in self.asks.keys():
                v1, v2 = self.asks[k], self.bids[k]
                self.ax1.plot(x, (np.array(v1) + np.array(v2)) / 2, label=k)
                self.ax1.fill_between(x=x, y1=v1, y2=v2, alpha=0.3)
            for i, k in enumerate(self.market_makers_cash.keys()):
                v = self.market_makers_cash[k]
                self.ax2.plot(x, v, label=k)
            for i, k in enumerate(self.dealers_cash.keys()):
                v = self.dealers_cash[k]
                self.ax3.plot(x, v, label=k)
            for i, k in enumerate(self.market_makers_portfolio.keys()):
                v = self.market_makers_portfolio[k]
                self.ax4.plot(x, v, label=k)
            self.ax1.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size':9})
            self.animation_fig.tight_layout()
            self.line1 = self.ax1.get_lines()
            self.line2 = self.ax2.get_lines()
            self.line3 = self.ax3.get_lines()
            self.line4 = self.ax4.get_lines()
        else:
            self.ax1.collections.clear()
            for i, k in enumerate(self.asks.keys()):
                v1, v2 = self.asks[k][-1], self.bids[k][-1]
                self.line1[i].set_data(x[-self.max_steps_animation:], np.append(self.line1[i].get_ydata(), [(v1 + v2) / 2])[-self.max_steps_animation:])
                self.ax1.fill_between(x[-self.max_steps_animation:], self.asks[k][-self.max_steps_animation:], self.bids[k][-self.max_steps_animation:], alpha=0.3)
                self.ax1.draw_artist(self.line1[i])
            self.ax1.relim()
            self.ax1.autoscale_view()
            for i, k in enumerate(self.market_makers_cash.keys()):
                v = self.market_makers_cash[k][-1:]
                self.line2[i].set_data(x[-self.max_steps_animation:], np.append(self.line2[i].get_ydata(), v)[-self.max_steps_animation:])
                self.ax2.draw_artist(self.line2[i])
            self.ax2.relim()
            self.ax2.autoscale_view()
            for i, k in enumerate(self.dealers_cash.keys()):
                v = self.dealers_cash[k][-1:]
                self.line3[i].set_data(x[-self.max_steps_animation:], np.append(self.line3[i].get_ydata(), v)[-self.max_steps_animation:])
                self.ax3.draw_artist(self.line3[i])
            self.ax3.relim()
            self.ax3.autoscale_view()
            for i, k in enumerate(self.market_makers_portfolio.keys()):
                v = self.market_makers_portfolio[k][-1:]
                self.line4[i].set_data(x[-self.max_steps_animation:], np.append(self.line4[i].get_ydata(), v)[-self.max_steps_animation:])
                self.ax4.draw_artist(self.line4[i])
            self.ax4.relim()
            self.ax4.autoscale_view()
            self.animation_fig.canvas.draw()
            self.animation_fig.canvas.flush_events()

    def plot_final(self):
        fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
        ax1 = fig.add_subplot(spec[0, :])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[2, 0])
        ax4 = fig.add_subplot(spec[1, 1])
        ax1.set_title('Evolution of Prices', y=1.20)
        ax1.set_xlabel('Time (steps)')
        ax1.set_ylabel('Price ($)')
        ax1.grid()
        ax2.set_title('Evolution of Market Makers Cash')
        ax2.set_ylabel('Cash ($)')
        ax2.grid()
        ax3.set_title('Evolution of Dealers Cash')
        ax3.set_xlabel('Time (steps)')
        ax3.set_ylabel('Cash ($)')
        ax3.grid()
        ax4.set_title('Evolution of Market Makers Portfolio')
        ax3.set_xlabel('Time (steps)')
        ax4.set_ylabel('Amount')
        ax4.grid()
        x = np.arange(0, self.max_steps, 1)
        for k in self.asks.keys():
            v1, v2 = self.asks[k], self.bids[k]
            ax1.plot(x, (np.array(v1) + np.array(v2)) / 2, label=k)
            ax1.fill_between(x=x, y1=v1, y2=v2, alpha=0.3)
        for k in self.market_makers_cash.keys():
            ax2.plot(x, self.market_makers_cash[k], label=k)
        for k in self.dealers_cash.keys():
            ax3.plot(x, self.dealers_cash[k], label=k)
        for k in self.market_makers_portfolio.keys():
            ax4.plot(x, self.market_makers_portfolio[k], label=k)
        ax1.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size':9})
        fig.tight_layout()
        plt.show()


    def __str__(self):
        to_print = ''
        for id_company in self.companies.keys():
            to_print += "%s | Ask: $%.2f | Bid: $%.2f\n" % (id_company.ljust(5), self.bids[id_company][-1], self.bids[id_company][-1])
        return to_print


class Company(Agent):
    """
    This class is used to represent a Company. It inherits
    from the Agent class. The company stocks are traded at a certain
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
