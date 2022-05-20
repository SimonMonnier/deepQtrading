from asyncore import read
import re
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

BUY = 0
SELL = 1
HOLD = 2


class TradingOpenEnv(gym.Env):
    def __init__(self, dataset_path, spread, period, nlot, episode_lenght=1):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiBinary(104)

        self.spread = spread
        self.period = period
        self.dataset = self._init_dataset(dataset_path, episode_lenght)
        self.nlot = nlot * 100000
        self.trade = False

        self.nstep = 0
        self.max_step = (756 * episode_lenght)
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
        self.nstep = 0
        self.episode_data = self._get_dataset_sample()
        self.state = self.episode_data[0:self.period]
        self.info = []
        return np.array(self.state).reshape(-1)

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

    def step(self, action):
        done = False
        reward = 0
        self.nstep += 1

        if len(self.episode_data) <= self.nstep + self.period or self.nstep == 100:
            done = True

        price = self.state[-1][DFOPEN]

        # TODO try with stop loss

        sl_buy, sl_sell = 0, 0
        profit_buy, profit_sell = 0, 0
        for i in range(4):
            next_state = self.episode_data[self.nstep+i]
            profit_buy += (next_state[DFOPEN] -
                           (price + self.spread))
            profit_sell += ((price + self.spread) -
                            next_state[DFOPEN])
            if next_state[DFLOW] <= price - (self.spread * 4):
                sl_buy += 1
            if next_state[DFHIGH] >= price + (self.spread * 4):
                sl_sell += 1

        if action == BUY:
            reward = profit_buy

        if action == SELL:
            reward = profit_sell

        if action == HOLD:
            reward = 0
            stop_loss = 0
            if sl_buy > 0:
                stop_loss += (((self.spread * 6) * sl_buy))
            if sl_sell > 0:
                stop_loss += (((self.spread * 6) * sl_sell))
            reward = stop_loss
            if profit_sell < profit_buy:
                reward = profit_buy
            else:
                reward = profit_sell
            if stop_loss > reward:
                reward = 10
            else:
                reward = -(reward)

        # Do 1 step
        # Update state
        self.state = self.episode_data[self.nstep:(self.nstep+self.period)]

        info = {"r": reward, "l": self.step,
                "episode": 0, "is_success": done}

        return np.array(self.state).reshape(-1), reward, done, info
