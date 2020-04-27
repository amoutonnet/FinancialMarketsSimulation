import numpy as np
import random
from . import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
np.set_printoptions(suppress=True)


class Market():
    def __init__(
        self,
        Nm,
        Nd,
        T,
        p0,
        m0,
        c0,
        A0,
        S,
        W,
        L,
        M
    ):
        self.Nm = Nm
        self.Nd = Nd
        self.T = T
        self.p0 = p0
        self.m0 = m0
        self.c0 = c0
        self.A0 = A0
        self.S = S
        self.W = W
        self.L = L
        self.M = M
        self.summarize()
        self.setup_access_dict()
        self.setup_slices_dict()
        self.m_obs = np.empty((2 * self.Nm + 2 * self.Nd, self.W), dtype=np.float32)
        self.d_obs = np.empty((2 * self.Nm + 3 * self.Nm + 1, self.W), dtype=np.float32)

    def summarize(self):
        utils.print_to_output(message=["%s:%s" % (str(key), str(val)) for key, val in self.__dict__.items()], title='Environment Initialization')

    def setup_access_dict(self):
        self.access_dict = {}
        self.access_dict["m_asks"] = list(range(0, 2 * self.Nm, 2))
        self.access_dict["m_port"] = list(range(1, 2 * self.Nm, 2))
        self.access_dict["d_cash"] = list(range(2 * self.Nm + 3 * self.Nm, 2 * self.Nm + self.Nd * (3 * self.Nm + 1), 3 * self.Nm + 1))
        self.access_dict["d_port"] = []
        self.access_dict["d_exec"] = []
        self.access_dict["d_trades"] = []
        for did in range(self.Nd):
            self.access_dict["d_port"].append(list(range(2 * self.Nm + did * (3 * self.Nm + 1), 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3)))
            self.access_dict["d_trades"].append(list(range(2 * self.Nm + did * (3 * self.Nm + 1) + 1, 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3)))
            self.access_dict["d_exec"].append(list(range(2 * self.Nm + did * (3 * self.Nm + 1) + 2, 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3)))
        self.access_dict["d_startid"] = list(range(2 * self.Nm, 2 * self.Nm + self.Nd * (3 * self.Nm + 1), 3 * self.Nm + 1))

    def setup_slices_dict(self):
        self.slices_dict = {}
        self.slices_dict["m_asks"] = slice(0, 2 * self.Nm, 2)
        self.slices_dict["m_port"] = slice(1, 2 * self.Nm, 2)
        self.slices_dict["d_cash"] = slice(2 * self.Nm + 3 * self.Nm, None, 3 * self.Nm + 1)
        self.slices_dict["d_port"] = []
        self.slices_dict["d_exec"] = []
        self.slices_dict["d_trades"] = []
        for did in range(self.Nd):
            self.slices_dict["d_port"].append(slice(2 * self.Nm + did * (3 * self.Nm + 1), 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3))
            self.slices_dict["d_trades"].append(slice(2 * self.Nm + did * (3 * self.Nm + 1) + 1, 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3))
            self.slices_dict["d_exec"].append(slice(2 * self.Nm + did * (3 * self.Nm + 1) + 2, 2 * self.Nm + (did + 1) * (3 * self.Nm + 1) - 1, 3))
        self.slices_dict["m_obs_trades"] = []
        self.slices_dict["m_obs_exec"] = []
        for cid in range(self.Nm):
            self.slices_dict["m_obs_trades"].append(slice(2 * self.Nm + cid * 3 + 1, None, 3 * self.Nm + 1))
            self.slices_dict["m_obs_exec"].append(slice(2 * self.Nm + cid * 3 + 2, None, 3 * self.Nm + 1))

    def reset(self):
        self.histdata = np.zeros((2 * self.Nm + 3 * self.Nm * self.Nd + self.Nd, self.T + self.W + 1), dtype=np.float32)
        self.histdata[self.slices_dict["m_asks"], :self.W + 1] = self.A0
        self.histdata[self.slices_dict["m_port"], :self.W + 1] = self.m0
        for did in range(self.Nd):
            self.histdata[self.slices_dict["d_port"][did], :self.W + 1] = self.p0
            self.histdata[self.slices_dict["d_exec"][did], :self.W + 1] = 1
        self.histdata[self.slices_dict["d_cash"], :self.W + 1] = self.c0
        self.step = self.W

    def get_market_state(self, step):
        return self.histdata[:2 * self.Nm, step - self.W:step]

    def get_mm_random_actions(self):
        return np.clip(np.random.normal(self.histdata[self.slices_dict["m_asks"], self.step], 5, size=(self.Nm,)), utils.DELTA, self.M)

    def get_d_random_actions(self):
        p = np.array([[1 / (2 * self.L + 1)] * (2 * self.L + 1)] * self.Nm)
        c = p.cumsum(axis=1)
        actions = np.empty((self.Nd, self.Nm))
        for did in range(self.Nd):
            u = np.random.rand(len(c), 1)
            actions[did] = (u < c).argmax(axis=1) - self.L
        return actions

    def get_mm_obs(self, cid):
        self.m_obs[:2 * self.Nm] = np.roll(self.get_market_state(self.step), -2 * cid, axis=0)
        self.m_obs[2 * self.Nm::2] = self.histdata[self.slices_dict["m_obs_trades"][cid], self.step - self.W:self.step]
        self.m_obs[2 * self.Nm + 1::2] = self.histdata[self.slices_dict["m_obs_exec"][cid], self.step - self.W:self.step]
        return self.m_obs

    def get_mm_rew(self, obs):
        def get_buysell_diff():
            current_trades = obs[:, -1][self.Nm * 2::2]
            total_buy = np.sum(current_trades[current_trades > 0])
            total_sell = np.sum(current_trades[current_trades < 0])
            diff = abs(total_buy - total_sell)
            return total_sell * total_buy / (1 + diff)

        # def get_log_diff_prices():
        #     log_diff = np.log(DELTA + abs(obs[0, -1] - obs[0, -2]) / (obs[0, -2] + obs[0, -1] + 1))
        #     return log_diff

        def get_extreme_penalty():
            current_ask_price = obs[0, -1]
            log_pen = np.log(current_ask_price)
            if log_pen >= 0:
                return 0
            else:
                return log_pen

        return get_buysell_diff() + get_extreme_penalty()

    def get_all_mm(self):
        observations = []
        rewards = []
        for cid in range(self.Nm):
            obs = self.get_mm_obs(cid)
            observations.append(obs)
            rewards.append(self.get_mm_rew(obs))
        return np.array(observations), np.array(rewards)

    def get_d_obs(self, did):
        self.d_obs[:2 * self.Nm] = self.get_market_state(self.step + 1)
        self.d_obs[2 * self.Nm:] = self.histdata[self.access_dict["d_startid"][did]: self.access_dict["d_startid"][did] + 3 * self.Nm + 1, self.step - self.W:self.step]
        return self.d_obs

    def get_d_rew(self, obs):
        current_obs = obs[:, -1]
        last_obs = obs[:, -2]

        def get_wealth_diff():
            current_cash = current_obs[-1]
            last_cash = last_obs[-1]
            current_portfolio_value = np.dot(
                current_obs[:self.Nm * 2][::2],
                current_obs[self.Nm * 2:-1][::3].T
            )
            last_portfolio_value = np.dot(
                last_obs[:self.Nm * 2][::2],
                last_obs[self.Nm * 2:-1][::3].T
            )
            return (current_cash + current_portfolio_value) / (last_cash + last_portfolio_value) - 1

        def get_penalty_execution():
            return -np.sum(1 - current_obs[self.Nm * 2 + 2:-1:3]) * 0

        return get_wealth_diff() + get_penalty_execution()

    def get_all_d(self):
        observations = []
        rewards = []
        for did in range(self.Nd):
            obs = self.get_d_obs(did)
            observations.append(obs)
            rewards.append(self.get_d_rew(obs))
        return np.array(observations), np.array(rewards)

    def prepare_next(self):
        self.histdata[:, self.step + 1] = self.histdata[:, self.step]
        self.step += 1

    def get_hist(self):
        return self.histdata

    def fix_market(self, prices):
        assert(prices.shape[0] == self.Nm)
        self.histdata[:2 * self.Nm:2, self.step] = prices

    def pass_orders(self, trades):
        assert(trades.shape[0] == self.Nd and trades.shape[1] == self.Nm)
        for did in range(self.Nd):
            self.histdata[self.slices_dict["d_trades"][did], self.step] = trades[did]

    def process_orders(self):
        for cid in range(self.Nm):
            buy_ids = []
            ask = self.histdata[self.access_dict["m_asks"][cid], self.step]
            bid = ask - self.S
            for did in range(self.Nd):
                portfolio = self.histdata[self.access_dict["d_port"][did][cid], self.step]
                trade = self.histdata[self.access_dict["d_trades"][did][cid], self.step]
                if trade < 0:
                    if portfolio >= -trade:
                        total = trade * bid
                        self.histdata[self.access_dict["m_port"][cid], self.step] -= trade
                        self.histdata[self.access_dict["d_port"][did][cid], self.step] += trade
                        self.histdata[self.access_dict["d_cash"][did], self.step] -= total
                        self.histdata[self.access_dict["d_exec"][did][cid], self.step] = 1
                    else:
                        self.histdata[self.access_dict["d_exec"][did][cid], self.step] = 0
                elif trade == 0:
                    self.histdata[self.access_dict["d_exec"][did][cid], self.step] = 1
                else:
                    buy_ids.append((did, trade))
            for did, trade in random.sample(buy_ids, len(buy_ids)):
                total = trade * ask
                cash = self.histdata[self.access_dict["d_cash"][did], self.step]
                if total <= cash and self.histdata[self.access_dict["m_port"][cid], self.step] >= trade:
                    self.histdata[self.access_dict["m_port"][cid], self.step] -= trade
                    self.histdata[self.access_dict["d_port"][did][cid], self.step] += trade
                    self.histdata[self.access_dict["d_cash"][did], self.step] -= total
                    self.histdata[self.access_dict["d_exec"][did][cid], self.step] = 1
                else:
                    self.histdata[self.access_dict["d_exec"][did][cid], self.step] = 0

    def create_figure(self):
        self.animation_fig = plt.figure(figsize=(10, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=self.animation_fig)
        self.axes, self.lines = [None] * 5, []
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

    def show_env(self, title='Simulation Visualisation'):
        self.create_figure()
        x = np.arange(0, self.T, 1)
        for cid in range(self.Nm):
            self.axes[0].plot(x, self.histdata[self.access_dict["m_asks"][cid], self.W:-1], alpha=0.8)
            self.axes[2].plot(x, self.histdata[self.access_dict["m_port"][cid], self.W:-1], alpha=0.8)
        for did in range(self.Nd):
            cash = self.histdata[self.access_dict["d_cash"][did], self.W:-1]
            portfolio_value = np.sum(self.histdata[self.slices_dict["d_port"][did], self.W:-1] * self.histdata[self.slices_dict["m_asks"], self.W:-1], axis=0)
            self.axes[1].plot(x, cash, alpha=0.8)
            self.axes[3].plot(x, portfolio_value, alpha=0.8)
            self.axes[4].plot(x, cash + portfolio_value, alpha=0.8)
        # self.axes[0].legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 9})
        for ax in self.axes:
            ax.set_ylim([min(0, ax.get_ylim()[0]), 1.1 * ax.get_ylim()[1]])
        self.animation_fig.tight_layout()
        self.animation_fig.subplots_adjust(top=0.87)
        self.animation_fig.suptitle(title, fontsize=15, x=0.54)
        plt.show()
