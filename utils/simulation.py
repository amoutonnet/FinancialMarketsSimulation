# Importing libraries for calculations
import numpy as np
import pandas as pd
import random
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
        if title is not None:
            print('\n' + title.center(70, '-')+'\n')
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

    def get_carac(self, carac, past=False):
        if past:
            return self.market.window_data[self.market.data_idx['%s_%d' % (carac, self.id_)]][-2]
        else:
            return self.market.window_data[self.market.data_idx['%s_%d' % (carac, self.id_)]][-1]

    def set_carac(self, carac, value):
        self.market.window_data[self.market.data_idx['%s_%d' % (carac, self.id_)]][-1] = value

    def add_to_carac(self, carac, value_to_add):
        self.market.window_data[self.market.data_idx['%s_%d' % (carac, self.id_)]][-1] += value_to_add

class MarketMakerAgent(Agent):
    """
    This class is used to represent a market maker type of agent. It inherits
    from the Agent class. The market maker operates for a company,
    by fixing the ask and the bid price. He tries to find the
    supply and demand equilibrium.
    """

    def __init__(self, id_, name, market, short_name, initial_price):
        super().__init__(id_, name, market)
        self.short_name = short_name

    def sample_action(self):
        return self.get_carac('ask_price') + np.random.normal(scale=5)

    def make_market(self, action):
        """ This function is called at each time step and
        allow the market maker to take a position on the
        market """
        self.set_carac('ask_price', action)

    def get_observation(self):
        return self.market.window_data[self.id_*4:(self.id_+1)*4]

    def __str__(self):
        return self.name

    __repr__ = __str__


class DealerAgent(Agent):
    """
    This class is used to represent a Dealer type of agent. It inherits
    from the Agent class. The dealer starts the game with
    a set of stocks, and an initial amount of cash and will try
    to maximize its profit by trading stocks.
    """

    def __init__(self, id_, name, market, initial_budget, initial_shares, initial_price):
        super().__init__(id_, name, market)

    def sample_action(self):
        # Random Action
        type_of_transaction = np.random.randint(3)
        if type_of_transaction == 0:
            id_mm = np.random.randint(self.market.nb_companies)
            amount = np.random.randint(1, MAX_AMOUNT)
        elif type_of_transaction == 1:
            id_mm = np.random.randint(self.market.nb_companies)
            inside_portfolio = self.get_carac('portfolio_dealer_%d' % id_mm)
            if inside_portfolio > 1:
                amount = np.random.randint(1, inside_portfolio)
            else:
                amount = inside_portfolio
        else:
            id_mm = -1
            amount = 0
        return [type_of_transaction, id_mm, amount]


    def trade(self, action):
        """ This function is called at each time step and
        allow the dealer to either buy a stock at the market
        price, sell it at the market price, or do nothing
        depending on the action he decides to take """
        self.set_carac('type_of_action', action[0])
        self.set_carac('traded_company', action[1])
        self.set_carac('amount_traded', action[2])
        if self.get_carac('type_of_action') == 0:
            self.market.buying_orders += [BuyingOrder(self.market, self, self.get_carac('traded_company'))]
        elif self.get_carac('type_of_action') == 1:
            self.market.selling_orders += [SellingOrder(self.market, self, self.get_carac('traded_company'))]
        else:
            self.set_carac('success', -1)

    def get_portfolio(self):
        start_idx = self.market.data_idx['portfolio_dealer_0_%d' % self.id_]
        return self.market.window_data[start_idx : start_idx + self.market.nb_companies][:, -1]

    def process_wealth(self):
        self.set_carac('portfolio_value', np.dot(self.get_portfolio().T, self.market.get_current_ask_prices()))
        self.set_carac('global_wealth', self.get_carac('portfolio_value') + self.get_carac('cash_dealer'))

    def get_observation(self):
        size = self.market.nb_companies + len(self.market.dealers_caracs)
        mask = [True, True, False, False]*self.market.nb_companies + [False]*size*self.id_ + [True]*size + [False]*size*(self.market.nb_dealers-self.id_-1)
        return self.market.window_data[mask]

    def __str__(self):
        return self.name.ljust(11, ' ')

    __repr__ = __str__


class Order():
    """ This class is used to represent an order passed by a dealer """

    def __init__(self, market, contracted_by, id_company):
        self.market = market
        self.contracted_by = contracted_by
        self.market_maker = self.market.market_makers[id_company]
        self.type_ = contracted_by.get_carac('type_of_action')
        self.amount = contracted_by.get_carac('amount_traded')

    def fulfill_order(self, verbose=False):
        """ This function is called when we want to fulfill this order """
        print_to_output('♦ New order processed: %s' % str(self), verbose=verbose)
        fully_fulfilled, cannot_process = self.process(verbose)
        if fully_fulfilled:
            print_to_output('Order was fully fullfilled', verbose=verbose)
            self.contracted_by.set_carac('success', 2)
        elif cannot_process:
            print_to_output('Order was not fullfilled because the dealer action is absurd', verbose=verbose)
            self.contracted_by.set_carac('success', 0)
        else:
            print_to_output('Order was not fullfilled at all due to lack of supply', verbose=verbose)
            self.contracted_by.set_carac('success', 1)

    def process(self, verbose):
        raise(NotImplementedError)

    def __str__(self):
        action = 'sell' if self.type_ else 'buy'
        return '%s wants to %s %d %s stocks' % (self.contracted_by, action, self.amount, self.market_maker)

    __repr__ = __str__


class SellingOrder(Order):
    """ This class inheriting from order is used to represent an immediate selling order,
    i.e. an order to be fulfilled during this time step at market price."""

    def __init__(self, market, contracted_by, id_company):
        super().__init__(market, contracted_by, id_company)

    def process(self, verbose):
        """ This function is called to process the order """
        fully_fulfilled, cannot_process = False, False
        if self.amount > 0:
            nb_stock_transfered = self.amount
            total_of_transaction = nb_stock_transfered * (self.market_maker.get_carac('ask_price') - self.market.spread)
            self.contracted_by.add_to_carac('cash_dealer', total_of_transaction)
            self.contracted_by.add_to_carac('portfolio_dealer_%d' % self.market_maker.id_, -nb_stock_transfered)
            self.market_maker.add_to_carac('portfolio', nb_stock_transfered)
            self.market_maker.add_to_carac('purchases', 1)
            self.amount -= nb_stock_transfered
            print_to_output('Details: %s just sold %d %s stocks to the market maker for %.2f' % (self.contracted_by, nb_stock_transfered, self.market_maker.short_name, total_of_transaction), verbose=verbose)
            if self.amount == 0:
                fully_fulfilled = True
        else:
            cannot_process = True
        return fully_fulfilled, cannot_process

class BuyingOrder(Order):
    """ This class inheriting from order is used to represent an immediate buying order,
    i.e. an order to be fulfilled during this time step at market price."""

    def __init__(self, market, contracted_by, id_company):
        super().__init__(market, contracted_by, id_company)

    def process(self, verbose):
        """ This function is called to process the order """
        fully_fulfilled, cannot_process = False, False
        cash_limit = self.contracted_by.get_carac('cash_dealer') // self.market_maker.get_carac('ask_price')
        if self.market_maker.get_carac('portfolio') > 0 and cash_limit > 0:
            nb_stock_transfered = min(self.amount, self.market_maker.get_carac('portfolio'), cash_limit)
            total_of_transaction = nb_stock_transfered * self.market_maker.get_carac('ask_price')
            self.contracted_by.add_to_carac('cash_dealer', -total_of_transaction)
            self.contracted_by.add_to_carac('portfolio_dealer_%d' % self.market_maker.id_, nb_stock_transfered)
            self.market_maker.add_to_carac('portfolio', -nb_stock_transfered)
            self.market_maker.add_to_carac('sales', 1)
            self.amount -= nb_stock_transfered
            print_to_output('Details: %s just bought %d %s stocks to the market maker for %.2f' % (self.contracted_by, nb_stock_transfered, self.market_maker.short_name, total_of_transaction), verbose=verbose)
            if self.amount == 0:
                fully_fulfilled = True
        else:
            cannot_process = True
        return fully_fulfilled, cannot_process

class Market():
    """
    This class is used to represent a market, with a
    certain number of companies, market makers and
    dealers evolving in it
    """

    def __init__(self, nb_companies, initial_nb_shares, nb_dealers, initial_dealer_budget, initial_price, window_size, spread, verbose=0, follow_hist_data=False):
        assert(nb_dealers < initial_nb_shares and initial_nb_shares % nb_dealers == 0)
        self.market_makers = {}       # A dictionnary containing market makers
        self.dealers = {}             # A dictionnary containing dealers
        self.companies = []
        self.buying_orders = []
        self.selling_orders = []
        self.nb_companies = nb_companies
        self.nb_dealers = nb_dealers
        self.spread = spread
        self.market_makers_caracs = [
            (initial_price, 'ask_price'),
            (0, 'portfolio'),
            (0, 'sales'),
            (0, 'purchases')
        ]
        self.dealers_caracs = [
            (initial_dealer_budget, 'cash_dealer'),
            ((initial_nb_shares // nb_dealers)*initial_price*self.nb_companies, 'portfolio_value'),
            (initial_dealer_budget+(initial_nb_shares // nb_dealers)*initial_price*self.nb_companies, 'global_wealth'),
            (-1, 'traded_company'),
            (2, 'type_of_action'),
            (0, 'amount_traded'),
            (-1, 'success')
        ]
        self.window_data = np.zeros((nb_dealers * (nb_companies + len(self.dealers_caracs)) + nb_companies * len(self.market_makers_caracs), window_size+1))

        self.data_idx = {'max_count': 0}
        self.window_size = window_size
        self.follow_hist_data = follow_hist_data
        self.max_steps = window_size
        self.size_of_animation = 0
        self.create_market_makers('init.csv', nb_companies, initial_nb_shares, initial_price, verbose)
        self.create_dealers(nb_dealers, initial_dealer_budget, initial_nb_shares // nb_dealers, initial_price, verbose)
        self.initial_window_data = self.window_data.copy()

        if self.follow_hist_data:
            self.historical_data = np.zeros((nb_dealers * (nb_companies + len(self.dealers_caracs)) + nb_companies * len(self.market_makers_caracs), 1000))
            self.historical_data[:, 0:self.window_size] = self.window_data[:, :-1]

    def create_market_makers(self, file, nb_companies, initial_nb_shares, initial_price, verbose):
        """ This function is used to create a market makers and its associated market maker within the market """
        company_data = pd.read_csv(file)
        assert(nb_companies > 0 and nb_companies < len(company_data) and initial_nb_shares > 0)
        print_to_output(title='Created Companies and Market Makers', verbose=verbose)
        for (_, j) in company_data[:nb_companies].iterrows():
            name = j[0]
            short_name = j[1] ####
            if short_name not in self.companies:
                self.companies += [short_name]
                id_mm = len(self.market_makers)
                self.market_makers[id_mm] = MarketMakerAgent(id_mm, 'Market Maker %s' % short_name, self, short_name, initial_price)
                for idx, carac in enumerate(self.market_makers_caracs):
                    self.data_idx['%s_%d' % (carac[1], id_mm)] = self.data_idx['max_count'] + idx
                    self.window_data[self.data_idx['%s_%d' % (carac[1], id_mm)]] = carac[0]
                self.data_idx['max_count'] += len(self.market_makers_caracs)
                print_to_output('%s (%s)' % (name, short_name), verbose=verbose)
            else:
                print_to_output('You tried to create two identical companies', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
                sys.exit()

    def create_dealers(self, nb_dealers, initial_dealer_budget, initial_shares, initial_price, verbose):
        assert(nb_dealers > 0 and initial_dealer_budget > 0 and initial_shares > 0)
        self.dealers_initial_cash = initial_dealer_budget
        for id_ in range(nb_dealers):
            self.dealers[id_] = DealerAgent(id_, 'Dealer ID%s' % id_, self, initial_dealer_budget, initial_shares, initial_price)
            for id_mm in self.market_makers:
                self.data_idx['portfolio_dealer_%d_%d' % (id_mm, id_)] = self.data_idx['max_count'] + id_mm
                self.window_data[self.data_idx['portfolio_dealer_%d_%d' % (id_mm, id_)]] = initial_shares
            self.data_idx['max_count'] += self.nb_companies
            for idx, carac in enumerate(self.dealers_caracs):
                self.data_idx['%s_%d' % (carac[1], id_)] = self.data_idx['max_count'] + idx
                self.window_data[self.data_idx['%s_%d' % (carac[1], id_)]] = carac[0]
            self.data_idx['max_count'] += len(self.dealers_caracs)
        print_to_output('%d dealers have been created' % nb_dealers, 'Created Dealers', verbose=verbose)

    def get_current_ask_prices(self):
        return self.window_data[0:self.nb_companies*4:4][:, -1]

    def prepare_next_step(self):
        """ This function is used between each step to prepare placeholders
        to track the state of the market """
        self.window_data[:, :-1] = self.window_data[:, 1:]
        self.window_data[2:self.nb_companies*4:4, -1] = 0
        self.window_data[3:self.nb_companies*4:4, -1] = 0
        if self.follow_hist_data:
            if self.max_steps < 1000:
                self.historical_data[:, self.max_steps] = self.window_data[:, -2]
            else:
                self.historical_data[:, :-1] = self.historical_data[:, 1:]
                self.historical_data[:, -1] = self.window_data[:, -2]
        self.max_steps += 1

    def settle_positions(self, actions, verbose):
        if len(actions) != self.nb_companies:
            print_to_output('Every market maker should have an action to do, your action dictionnary is incomplete', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
            sys.exit()
        for i in range(len(actions)):
            self.market_makers[i].make_market(actions[i])
        for dealer in self.dealers.values():
            dealer.process_wealth()
        print_to_output(self, 'Positions of Market Makers', overoneline=False, verbose=verbose>0)

    def settle_trading(self, actions, verbose):
        if len(actions) != self.nb_dealers:
            print_to_output('Every dealer should have an action to do, your action dictionnary is incomplete', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')
            sys.exit()
        shuffled_actions = random.sample(list(enumerate(actions)), len(actions))
        print_to_output(title='Orders', verbose=verbose>1)
        for id_d, action in shuffled_actions:
            self.dealers[id_d].trade(action)
        while len(self.selling_orders):
            self.selling_orders.pop().fulfill_order(verbose > 1)
        while len(self.buying_orders):
            self.buying_orders.pop().fulfill_order(verbose > 1)

    def reset(self):
        self.window_data = self.initial_window_data.copy()
        if self.follow_hist_data:
            self.historical_data[:, 0:self.window_size] = self.window_data[:, :-1]

    def create_figure(self):
        self.animation_fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=self.animation_fig)
        self.axes = [None]*5
        self.lines = []
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
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value'), (4,'global_wealth')]:
                    self.axes[i].plot(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1], alpha=0.8)
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
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value'), (4,'global_wealth')]:
                    self.lines[i][id_d].set_data(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1])
                    self.axes[i].draw_artist(self.lines[i][id_d])
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()
            self.animation_fig.canvas.draw()
            self.animation_fig.canvas.flush_events()

    def plot_final(self):
        if self.follow_hist_data:
            self.create_figure()
            x = np.arange(0, min(self.max_steps, 1000), 1)
            for id_mm in self.market_makers:
                for i,j in [(0,'ask_price'), (2,'portfolio')]:
                    self.axes[i].plot(x, self.historical_data[self.data_idx['%s_%d' % (j, id_mm)]][:len(x)], label=self.market_makers[id_mm].short_name, alpha=0.8)
            for id_d in self.dealers:
                for i,j in [(1,'cash_dealer'), (3,'portfolio_value'), (4,'global_wealth')]:
                    self.axes[i].plot(x, self.historical_data[self.data_idx['%s_%d' % (j, id_d)]][:len(x)])
            self.axes[0].legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 9})
            for ax in self.axes:
                ax.set_ylim([min(0, ax.get_ylim()[0]), ax.get_ylim()[1]])
            self.animation_fig.tight_layout()
            plt.show()
        else:
            print_to_output('You decided to not follow historical data, therefore you cannot vizualise it', '/!\\/!\\/!\\ERROR/!\\/!\\/!\\')

    def __str__(self):
        to_print = ''
        for id_mm in self.market_makers:
            to_print += "%s | Ask: $%.2f | Bid: $%.2f\n" % (self.market_makers[id_mm].short_name.ljust(5), self.market_makers[id_mm].get_carac('ask_price'), self.market_makers[id_mm].get_carac('ask_price') - self.spread)
        return to_print
