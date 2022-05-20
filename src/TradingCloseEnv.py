from asyncore import read
from tracemalloc import stop
import numpy as np
import pandas as pd
import random
import time
import plotly.graph_objects as go
import gym
from gym import Env, spaces, utils

DFOPEN = 0
DFHIGH = 1
DFLOW = 2
DFCLOSE = 3

#
# dataset_path : path of dataset.csv
# spread : brooker fee on transaction
# period : Number of ticks (current and previous ticks) returned as a observation.
# sold : start sold
# min_sold : minimun sold where game stop if reached
# episode_lenght :  duration of an episode in days
#

CLOSE = 0
HOLD = 1


class TradingCloseEnv(gym.Env):
    def __init__(self, dataset_path, spread, period, nlot, episode_lenght=1):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(106)

        self.spread = spread
        self.period = period
        self.dataset = self._init_dataset(dataset_path, episode_lenght)
        self.nlot = nlot * 100000
        self.trade = True
        self.position = 0

        self.buy_price = None
        self.sell_price = None
        self.nstep = 0
        self.max_step = 756 * episode_lenght
        self.episode_data = self._get_dataset_sample()
        self.state = self.episode_data[self.nstep:period]

        self.info = []

    def _get_dataset_sample(self):
        segment = self.dataset[random.randrange(0, len(self.dataset))]
        segment = segment.drop(
            columns=['Date', 'Time', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh'])
        segment = np.array(segment)
        return segment

    def _init_dataset(self, dataset_path, episode_lenght):
        df = pd.read_csv(dataset_path, sep=',')
        size = df.shape[0]
        # min_by_day = 1440 * episode_lenght for min
        #split_size = size / min_by_day
        hour_by_day = (756 * episode_lenght)
        split_size = size / hour_by_day
        split_dataset = np.array_split(df, split_size)
        return split_dataset

    def reset(self):
        self.nstep = 1
        self.episode_data = self._get_dataset_sample()
        self.state = self.episode_data[0:self.period]
        self.buy_price, self.sell_price = None, None
        profit_buy, profit_sell = 0, 0
        price = self.state[-1][DFOPEN]
        for i in range(4):
            next_state = self.episode_data[self.nstep+i]
            profit_buy += (next_state[DFOPEN] -
                           (price + self.spread))
            profit_sell += ((price + self.spread) -
                            next_state[DFOPEN])
        if profit_buy > profit_sell:
            self.position = 1
            self.buy_price = self.state[-1][DFOPEN]
            self.state = self.episode_data[self.nstep:self.period+self.nstep]
            profit = (self.state[-1][DFOPEN] -
                      (self.buy_price + self.spread))
            more_param = np.array([self.position, profit])
        else:
            self.position = 2
            self.sell_price = self.state[-1][DFOPEN]
            self.state = self.episode_data[self.nstep:self.period+self.nstep]
            profit = ((self.sell_price - self.spread) -
                      self.state[-1][DFOPEN])
            more_param = np.array([self.position, profit])
        self.info = []
        return np.array(np.append(self.state, more_param))
        # return np.array(self.state)

    def render(self):
        df = pd.DataFrame(self.state)
        fig = go.Figure(data=[go.Candlestick(
            open=np.array(df[DFOPEN]),
            high=np.array(df[DFHIGH]),
            low=np.array(df[DFLOW]),
            close=np.array(df[DFCLOSE]))
        ])
        fig.show()

        print('*** Game stats ***\nTrade: '+str(self.trade))
        print('Buy price: {0}\tSell price: {1}'.format(
            self.buy_price, self.sell_price))
        print('Current price: {0}'.format(self.state[-1][DFCLOSE]))
        print('Profit: '+str(self._get_profit()))
        print('Sold: '+str(self.sold))
        print('Trade sold: {0}'.format(self.trade_sold))

    def _get_profit(self, candle=DFCLOSE):
        profit = 0
        price = self.state[-1][candle]
        if self.trade and self.buy_price != None:
            profit = price - (self.buy_price + self.spread)
        if self.trade and self.sell_price != None:
            profit = (self.sell_price - self.spread) - price
        return profit

    def step(self, action):
        done = False
        profit = self._get_profit()
        self.nstep += 1

        if len(self.episode_data) <= self.nstep + self.period or self.nstep == 100:
            done = True
            action = CLOSE

        # TODO maybe reward + when stop loss

        # Close action
        if self.trade and action == CLOSE:
            reward = profit
            done = True

        # Update state
        self.state = self.episode_data[self.nstep:(self.nstep+self.period)]

        # Holde action
        if action == HOLD:
            if self.buy_price != None:
                reward = (self._get_profit() - profit)
            else:
                reward = (profit - self._get_profit())

        info = {"r": reward, "l": self.nstep,
                "episode": 0, "is_success": done}

        more_param = np.array([self.position, profit])

        return np.array(np.append(self.state, more_param)), reward, done, info
