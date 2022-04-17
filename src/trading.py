import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
from gym import spaces
# import mplfinance as mpf
# import cv2

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
CLOSE = 2
HOLD = 3


class TradingEnv():
    def __init__(self, dataset_path, spread, period, sold, min_sold, nlot, episode_lenght=1):
        self.action_space = spaces.Discrete(4)

        self.spread = spread
        self.period = period
        self.dataset = self._init_dataset(dataset_path, episode_lenght)
        self.sold = sold
        self.trade_sold = self.sold
        self.start_sold = sold
        self.nlot = nlot * 100000
        self.min_sold = min_sold
        self.trade = False

        self.buy_price = None
        self.sell_price = None
        self.nstep = 0
        self.max_step = 300 * episode_lenght
        self.episode_data = self._get_dataset_sample()
        self.state = self.episode_data[self.nstep:period]
        self.total_trade = 0

    def _get_dataset_sample(self):
        segment = self.dataset[random.randrange(0, len(self.dataset))]
        segment = segment.drop(
            columns=['Date', 'Time', 'BCh', 'AO', 'AH', 'AL', 'AC', 'ACh'])
        segment = np.array(segment)
        return segment

    def _init_dataset(self, dataset_path, episode_lenght):
        df = pd.read_csv(dataset_path, sep=',')
        size = df.shape[0]
        min_by_day = 1440 * episode_lenght
        split_size = size / min_by_day
        split_dataset = np.array_split(df, split_size)
        return split_dataset

    def reset(self):
        self.buy_price = None
        self.sell_price = None
        self.sold = self.start_sold
        self.trade_sold = self.sold
        self.nstep = 0
        self.total_trade = 0
        self.episode_data = self._get_dataset_sample()
        self.state = self.episode_data[0:self.period]
        state = np.append(np.append(np.append(self.state, 0),0),0)
        
        return state

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

    def _get_profit(self):
        profit = 0
        price = self.state[-1][DFCLOSE]
        if self.trade and self.buy_price != None:
            profit = price - (self.buy_price + self.spread)
        if self.trade and self.sell_price != None:
            profit = (self.sell_price - self.spread) - price
        return profit

    def step(self, action):
        done = False
        reward = 0
        self.nstep += 1

        if len(self.episode_data) <= self.nstep + self.period:
            done = True

            if self.trade == True:
                action = CLOSE

        price = self.state[-1][DFCLOSE]

        if self.trade and (action == SELL or action == BUY) and self._get_profit() <= 0:
            reward = -10
        if self.trade == False and action == CLOSE and self._get_profit() <= 0:
            reward = -10

        if self.trade == False and action == BUY:
            self.trade = True
            self.total_trade += 1
            self.buy_price = price
            reward = 10

        if self.trade == False and action == SELL:
            self.trade = True
            self.total_trade += 1
            self.sell_price = price
            reward = 10

        if action == HOLD and self.trade == True and self._get_profit() > 0:
            reward = 10
        elif action == HOLD and self.trade == True and self._get_profit() <= 0:
            reward = -10

        if action == HOLD and self.trade == False:
            reward = -10
        
        if self.trade == True and action == CLOSE and self._get_profit() > 0:
            reward = 100
            self.sold = (self._get_profit() * self.nlot) + self.sold
            self.trade_sold = self.sold
            self.buy_price = None
            self.sell_price = None
            self.trade = False
        elif self.trade == True and action == CLOSE and self._get_profit() <= 0:
            reward = 0
            self.sold = (self._get_profit() * self.nlot) + self.sold
            self.trade_sold = self.sold
            self.buy_price = None
            self.sell_price = None
            self.trade = False

        if self.trade_sold < self.min_sold or self.sold < self.min_sold:
            done = True
            self.sold = (self._get_profit() * self.nlot) + self.sold
            self.trade_sold = self.sold
            reward = -1000

        if (action == HOLD or action == BUY or action == SELL) and self.trade == True and self._get_profit() < 0:
            reward = reward - 100
        if (action == BUY or action == SELL) and self._get_profit() > 0:
            reward = reward + 10
        if (action == BUY or action == SELL) and self._get_profit() <= 0:
            reward = reward -10
        # Update state
        self.state = self.episode_data[self.nstep:(self.nstep+self.period)]

        # Update trade sold
        self.trade_sold = (self._get_profit() * self.nlot) + self.sold

        trade_state = 0
        if (self.trade == True and self.buy_price != None):
            trade_state = 1
            state = np.append(np.append(np.append(self.state, trade_state), self._get_profit()),self.buy_price)
        if (self.trade == True and self.sell_price != None):
            trade_state = 2
            state = np.append(np.append(np.append(self.state, trade_state), self._get_profit()), self.sell_price)
        else:
            state = np.append(np.append(np.append(self.state, trade_state), self._get_profit()),0)
        # df = pd.DataFrame(self.state)
        # mpf.plot(df, type='candle', style='charles', savefig='test-mplfiance.png')
        # img_path = r"C:\Users\smonn\Desktop\DeepTrading\src\test-mplfiance.png"

        # img = cv2.imread(img_path)
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # state = cv2.resize(img, (180,180))

        return state, reward, done