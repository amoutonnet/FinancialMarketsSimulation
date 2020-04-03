# Importing libraries for calculations
import numpy as np
import scipy.stats
import pandas as pd
import random
import copy
# Importing libraries for plots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')

# For Yida, uncomment the next line and comment the previous line

# matplotlib.use('Qt5Agg')

# Importing useful libraries
import sys
from tqdm import tqdm

MAX_AMOUNT = 7


def print_to_output(message=None, title=None, overoneline=True, verbose=True):
    """
    This function is used to simplify the printing task. You can
    print a message with a title, over one line or not for lists.
    """
    if verbose:
        # We print the title between '----'
        if title is not None:
            print('\n' + title.center(70, '-')+'\n')
        # We print the message
        if message is not None:
            # Lists have particular treatment
            if isinstance(message, list):
                # Either printed over one line
                if overoneline:
                    to_print = ''
                    for i in message:
                        to_print += '%s | ' % str(i)
                    print(to_print[:-3])
                # Or not
                else:
                    for i in message:
                        print(i)
            else:
                print(message)


class Agent():
    """
    This mother class defines the basic attributes of an
    Agent on the market. 
    """

    def __init__(self, id_, name, market):
        """
        This Agent will have as attributes an ID, a name,
        and the market he is a part of.
        """
        self.id_ = id_
        self.name = name
        self.market = market
        self.mask = None

    def get_charac(self, charac):
        """
        This function takes as input the characteristic of the agent we want to get
        inside the global array keeping track of everything on the market
        """
        return self.market.window_data[self.market.data_idx['%s_%d' % (charac, self.id_)]][-1]

    def set_charac(self, charac, value):
        """
        This function takes as input the characteristic of the agent we want to set
        inside the global array keeping track of everything on the market, and the value
        to set
        """
        self.market.window_data[self.market.data_idx['%s_%d' % (charac, self.id_)]][-1] = value

    def add_to_charac(self, charac, value_to_add):
        """
        This function is useful to add some values to an already set characteristic of the global
        array keeping track of everything on the market,
        """
        self.set_charac(charac, self.get_charac(charac) + value_to_add)

class MarketMakerAgent(Agent):
    """
    This class is used to represent a market maker type of agent. It inherits
    from the Agent class. The market maker operates for a company,
    by fixing the ask price. He tries to find the supply and demand equilibrium.
    """

    def __init__(self, id_, name, market, short_name):
        """
        In addition to the common attributes of an agent, market makers 
        also have as attribute the short name of the associated company.
        They also have a mask allowing them to get only the observable part of 
        the state of the environment from the global array describing the 
        whole market.
        """
        super().__init__(id_, name, market)
        self.short_name = short_name
        self.mask = self.id_*[False]*len(self.market.market_makers_characs) +\
                    [True]*len(self.market.market_makers_characs) +\
                    (self.market.nb_companies-self.id_-1)*[False]*len(self.market.market_makers_characs) +\
                    self.market.nb_dealers*(
                        self.id_*[False]*len(self.market.dealers_characs_per_company)+\
                        [False, True, True]+\
                        (self.market.nb_companies-self.id_-1)*[False]*len(self.market.dealers_characs_per_company)+\
                        [False]*len(self.market.dealers_characs)
                    )

    def take_action(self, mu=0, sigma=1):
        """
        The action he takes moves the market by a certain value. Concretely, he draws this value according to
        a gaussian distribution whose parameters he can manage. This gaussian distribution is truncated
        to avoid ask prices being negative.
        """
        return scipy.stats.truncnorm.rvs(a=(-self.get_charac('ask_price') - mu) / sigma, b=np.inf, loc=mu, scale=sigma, size=1)[0]

    def sample_action(self):
        """
        This function returns a sample of an action (i.e. a value drawn according to a truncated 
        gaussian distribution of mean 0 and scale 1)
        """
        return self.take_action()

    def make_market(self, action):
        """
        This function is called to process the action taken
        """
        self.add_to_charac('ask_price', action)

    def get_observation(self):
        """
        This function returns the part of the environment the agent has access to when taking a decision.
        """
        return self.market.window_data[self.mask]

    def __str__(self):
        return self.name

    __repr__ = __str__


class DealerAgent(Agent):
    """
    This class is used to represent a Dealer type of agent. 
    It inherits from the Agent class.
    """

    def __init__(self, id_, name, market):
        """
        The dealer has the same attributes as a common agent, plus a mask allowing to get only the
        observable part of the state of the environment from the global array describing the whole
        market, and a starting index corresponding to the index in the global array from which the
        data about the dealer starts.
        """
        super().__init__(id_, name, market)
        size = self.market.nb_companies*len(self.market.dealers_characs_per_company) + len(self.market.dealers_characs)
        self.mask = [True, True]*self.market.nb_companies +\
                    [False]*size*self.id_ +\
                    [True]*self.market.nb_companies*len(self.market.dealers_characs_per_company) + [True, False] +\
                    [False]*size*(self.market.nb_dealers-self.id_-1)

    def set_starting_data_idx(self):
        self.starting_data_idx = self.market.data_idx['portfolio_dealer_0_%d' % self.id_]

    def get_action(self, p=None):
        """
        The action he takes is basically choosing an amount to trade for each companies. This amount
        is drawn randomly between [|-MAX_AMOUNT, MAX_AMOUNT|], he can manage the weights (i.e. the 
        probability of drawing a particular number in this interval). It returns a 1D array with the
        amount of company i he wants to trade at index i
        """
        if p is None:
            p = np.array([[0.05]*MAX_AMOUNT+[0.3]+[0.05]*MAX_AMOUNT]*self.market.nb_companies)
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1) - MAX_AMOUNT

    def sample_action(self):
        """
        This function returns a sample of an action. For each company, the amount to trade is draw
        randomly with an equal chance of drawing something positive and negative, and a chance of 30%
        of drawing an amount of 0 (i.e. do not trade)
        """
        return self.get_action()

    def trade(self, actions):
        """
        This function is called to place the orders based on the action taken
        """
        for id_mm, act in enumerate(actions):
            self.set_charac('amount_traded_%d' % id_mm, act)
            if act > 0:
                # For a buying order we place it directly
                self.market.buying_orders += [(self.id_, id_mm, act)]
                return
            elif act < 0:
                # For a selling order, we place it only if we have enough stocks in our portfolio
                if self.get_charac('portfolio_dealer_%d' % id_mm) >= -act:
                    self.market.selling_orders += [(self.id_, id_mm, act)]
                    return
            self.set_charac('transaction_executed_%d' % id_mm, 0)

    def get_portfolio(self):
        """
        This function returns the state of the portfolio of the dealer, i.e. the number of stocks
        he owns for every companies as a 1D array.
        """
        return self.market.window_data[:,-1][self.starting_data_idx :\
                                             self.starting_data_idx + self.market.nb_companies*len(self.market.dealers_characs_per_company) :\
                                             len(self.market.dealers_characs_per_company)] 

    def get_observation(self):
        """
        This function returns the part of the environment the agent has access to when taking a decision.
        """
        return self.market.window_data[self.mask]

    def __str__(self):
        return self.name.ljust(11, ' ')

    __repr__ = __str__

class Market():
    """
    This class is used to represent a market, with a
    certain number of companies, market makers and
    dealers evolving in it
    """

    def __init__(self, nb_companies, day_length, initial_nb_shares, nb_dealers, initial_dealer_budget, initial_price, window_size, spread, verbose=0, follow_hist_data=False):
        self.market_makers = {}       # A dictionnary containing market makers
        self.dealers = {}             # A dictionnary containing dealers
        self.companies = []
        self.buying_orders = []
        self.selling_orders = []
        self.nb_companies = nb_companies
        self.nb_dealers = nb_dealers
        self.day_length = day_length
        self.spread = spread
        self.market_makers_characs = [
            (initial_price, 'ask_price'),
            (0, 'portfolio'),
        ]
        initial_nb_shares_per_dealer = initial_nb_shares // nb_dealers
        initial_portfolio_value = initial_nb_shares_per_dealer*initial_price*self.nb_companies
        self.dealers_characs_per_company = [
            (initial_nb_shares_per_dealer, 'portfolio_dealer'),
            (0, 'amount_traded'),
            (0, 'transaction_executed')
        ]
        self.dealers_characs = [
            (initial_dealer_budget, 'cash_dealer'),
            (initial_portfolio_value, 'portfolio_value'),
        ]

        assert(nb_dealers < initial_nb_shares and initial_nb_shares % nb_dealers == 0)
        assert(nb_dealers > 0 and initial_dealer_budget > 0 and initial_nb_shares_per_dealer > 0)

        self.data_size = nb_dealers * (nb_companies*len(self.dealers_characs_per_company) + len(self.dealers_characs)) +\
                         nb_companies * len(self.market_makers_characs)
        self.window_data = np.zeros((self.data_size, window_size+1))
        self.data_idx = {'max_count': 0}
        self.window_size = window_size
        self.follow_hist_data = follow_hist_data
        self.max_steps = window_size
        self.create_market_makers('init.csv', nb_companies, verbose)
        self.create_dealers(nb_dealers, verbose)
        self.initial_window_data = copy.deepcopy(self.window_data)
        if self.follow_hist_data:
            self.historical_data = np.zeros((self.data_size, day_length+window_size))
            self.historical_data[:, 0:window_size] = self.window_data[:, :-1]

    def create_market_makers(self, file, nb_companies, verbose):
        """ This function is used to create a market makers and its associated market maker within the market """
        company_data = pd.read_csv(file)
        assert(nb_companies > 0 and nb_companies < len(company_data))
        print_to_output(title='Created Companies and Market Makers', verbose=verbose)
        for (_, j) in company_data[:nb_companies].iterrows():
            name = j[0]
            short_name = j[1]
            if short_name not in self.companies:
                self.companies += [short_name]
                id_mm = len(self.market_makers)
                self.market_makers[id_mm] = MarketMakerAgent(id_mm, 'Market Maker %s' % short_name, self, short_name)
                for idx, charac in enumerate(self.market_makers_characs):
                    self.data_idx['%s_%d' % (charac[1], id_mm)] = self.data_idx['max_count'] + idx
                    self.window_data[self.data_idx['%s_%d' % (charac[1], id_mm)]] = charac[0]
                self.data_idx['max_count'] += len(self.market_makers_characs)
                print_to_output('%s (%s)' % (name, short_name), verbose=verbose)
            else:
                print_to_output('You tried to create two identical companies', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
                sys.exit()

    def create_dealers(self, nb_dealers, verbose):
        for id_ in range(nb_dealers):
            self.dealers[id_] = DealerAgent(id_, 'Dealer ID%s' % id_, self)
            for id_mm in self.market_makers:
                for idx, charac in enumerate(self.dealers_characs_per_company):
                    key = '%s_%d_%d' % (charac[1], id_mm, id_)
                    self.data_idx[key] = self.data_idx['max_count'] + idx
                    self.window_data[self.data_idx[key]] = charac[0]
                self.data_idx['max_count'] += len(self.dealers_characs_per_company)
            for idx, charac in enumerate(self.dealers_characs):
                self.data_idx['%s_%d' % (charac[1], id_)] = self.data_idx['max_count'] + idx
                self.window_data[self.data_idx['%s_%d' % (charac[1], id_)]] = charac[0]
            self.data_idx['max_count'] += len(self.dealers_characs)
            self.dealers[id_].set_starting_data_idx()
        print_to_output('%d dealers have been created' % nb_dealers, 'Created Dealers', verbose=verbose)

    def get_current_ask_prices(self):
        return self.window_data[:, -1][0:self.nb_companies*len(self.market_makers_characs):len(self.market_makers_characs)]

    def prepare_next_step(self):
        self.window_data[:, :-1] = self.window_data[:, 1:]
        if self.follow_hist_data:
            self.historical_data[:, self.max_steps] = self.window_data[:, -2]
        current_ask_prices = self.get_current_ask_prices()
        for id_d in range(self.nb_dealers):
            self.dealers[id_d].set_charac('portfolio_value', np.dot(self.dealers[id_d].get_portfolio(), current_ask_prices.T))
        self.max_steps += 1

    def settle_positions(self, actions, verbose):
        if len(actions) != self.nb_companies:
            print_to_output('Every market maker should have an action to do, your action dictionnary is incomplete', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
            sys.exit()
        for i in range(len(actions)):
            self.market_makers[i].make_market(actions[i])
        print_to_output(self, 'Positions of Market Makers', overoneline=False, verbose=verbose>0)

    def settle_trading(self, actions, verbose):
        if len(actions) != self.nb_dealers:
            print_to_output('Every dealer should have an action to do, your action dictionnary is incomplete', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
            sys.exit()
        print_to_output(title='Orders', verbose=verbose>1)
        for id_d, action in enumerate(actions):
            self.dealers[id_d].trade(action)
        while len(self.selling_orders):
            self.fulfill_order(*self.selling_orders.pop(), verbose > 1)
        random.shuffle(self.buying_orders)
        while len(self.buying_orders):
            self.fulfill_order(*self.buying_orders.pop(), verbose > 1)

    def fulfill_order(self, id_d, id_mm, amount, verbose=False):
        contracted_by = self.dealers[id_d]
        market_maker = self.market_makers[id_mm]
        if verbose:
            action = 'sell' if amount<0 else 'buy'
            print_to_output('♦ New order processed: %s wants to %s %d %s stocks' % (contracted_by, action, abs(amount), market_maker))
        if amount > 0:
            cash_limit = contracted_by.get_charac('cash_dealer') // market_maker.get_charac('ask_price')
            if cash_limit >= amount and market_maker.get_charac('portfolio') >= amount:
                total_of_transaction = amount * market_maker.get_charac('ask_price')
            else:
                print_to_output('Order was not executed', verbose=verbose)
                contracted_by.set_charac('transaction_executed_%d' % (market_maker.id_), 0)
                return
        elif amount < 0:
            total_of_transaction = amount * (market_maker.get_charac('ask_price') - self.spread)
        contracted_by.add_to_charac('cash_dealer', -total_of_transaction)
        contracted_by.add_to_charac('portfolio_dealer_%d' % id_mm, amount)
        market_maker.add_to_charac('portfolio', -amount)
        contracted_by.set_charac('transaction_executed_%d' % (id_mm), 1)
        if verbose:
            print_to_output('Order executed for %.2f' % abs(total_of_transaction))

    def reset(self):
        self.window_data = copy.deepcopy(self.initial_window_data)
        if self.follow_hist_data:
            self.historical_data[:, 0:self.window_size] = self.window_data[:, :-1]

    def create_figure(self):
        self.animation_fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=self.animation_fig)
        self.axes, self.lines = [None]*5, []
        self.axes[0] = self.animation_fig.add_subplot(spec[0, :])
        self.axes[1] = self.animation_fig.add_subplot(spec[1, 0])
        self.axes[2] = self.animation_fig.add_subplot(spec[2, 1])
        self.axes[3] = self.animation_fig.add_subplot(spec[1, 1])
        self.axes[4] = self.animation_fig.add_subplot(spec[2, 0])
        self.axes[0].set_title('Evolution of Prices', y=1.20)
        self.axes[0].set_ylabel('Price ($)')
        self.axes[1].set_title('Evolution of Dealers Cash')
        self.axes[1].set_ylabel('Cash ($)')
        self.axes[2].set_title('Evolution of Market Makers Portfolio')
        self.axes[2].set_ylabel('Amount')
        self.axes[3].set_title('Evolution of Dealers Portfolio Value')
        self.axes[3].set_ylabel('Amount')
        self.axes[4].set_title('Evolution of Dealers Global Wealth')
        self.axes[4].set_ylabel('Amount')
        for ax in self.axes:
            ax.grid()
            ax.set_xlabel('Time (Steps)')
            self.lines += [ax.get_lines()]

    def init_animation(self):
        self.create_figure()
        self.animation_started = True
        self.animation_fig.canvas.draw()
        plt.show(block=False)

    def step_animation(self):
        x = np.arange(self.max_steps - self.window_size, self.max_steps, 1)
        if self.max_steps == self.window_size:
            for id_mm in self.market_makers:
                for i,j in [(0,'ask_price'), (2,'portfolio')]:
                    self.axes[i].plot(x, self.window_data[self.data_idx['%s_%d' % (j, id_mm)]][:-1], label=self.market_makers[id_mm].short_name, alpha=0.8)
            for id_d in self.dealers:
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value')]:
                    self.axes[i].plot(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1], alpha=0.8)
                self.axes[4].plot(x, self.window_data[self.data_idx['cash_dealer_%d' % id_d]][:-1] + self.window_data[self.data_idx['portfolio_value_%d' % id_d]][:-1], alpha=0.8)
            self.axes[0].legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 9})
            self.animation_fig.tight_layout()
            for i in range(len(self.lines)):
                self.lines[i] = self.axes[i].get_lines()
        else:
            for id_mm in self.market_makers:
                for i,j in [(0,'ask_price'), (2,'portfolio')]:
                    self.lines[i][id_mm].set_data(x, self.window_data[self.data_idx['%s_%d' % (j, id_mm)]][:-1])
                    self.axes[i].draw_artist(self.lines[i][id_mm])
            for id_d in self.dealers:
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value')]:
                    self.lines[i][id_d].set_data(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1])
                    self.axes[i].draw_artist(self.lines[i][id_d])
                self.lines[4][id_d].set_data(x, self.window_data[self.data_idx['cash_dealer_%d' % id_d]][:-1] + self.window_data[self.data_idx['portfolio_value_%d' % id_d]][:-1])
                self.axes[4].draw_artist(self.lines[i][id_d])
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()
            self.animation_fig.canvas.draw()
            self.animation_fig.canvas.flush_events()

    def plot_final(self):
        if self.follow_hist_data:
            self.create_figure()
            x = np.arange(0, self.day_length, 1)
            for id_mm in self.market_makers:
                for i,j in [(0,'ask_price'), (2,'portfolio')]:
                    self.axes[i].plot(x, self.historical_data[self.data_idx['%s_%d' % (j, id_mm)]][self.window_size:], label=self.market_makers[id_mm].short_name, alpha=0.8)
            for id_d in self.dealers:
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value')]:
                    self.axes[i].plot(x, self.historical_data[self.data_idx['%s_%d' % (j, id_d)]][self.window_size:], alpha=0.8)
                self.axes[4].plot(x, self.historical_data[self.data_idx['cash_dealer_%d' % id_d]][self.window_size:] + self.historical_data[self.data_idx['portfolio_value_%d' % id_d]][self.window_size:], alpha=0.8)
            self.axes[0].legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 9})
            for ax in self.axes:
                ax.set_ylim([min(0, ax.get_ylim()[0]), ax.get_ylim()[1]])
            self.animation_fig.tight_layout()
            plt.show()
        else:
            print_to_output('You decided not to follow historical data, therefore you cannot vizualise it', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')

    def __str__(self):
        to_print = ''
        for id_mm in self.market_makers:
            to_print += "%s | Ask: $%.2f | Bid: $%.2f\n" % (self.market_makers[id_mm].short_name.ljust(5), self.market_makers[id_mm].get_charac('ask_price'), self.market_makers[id_mm].get_charac('ask_price') - self.spread)
        return to_print
