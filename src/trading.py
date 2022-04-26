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
        self.max_step = 1640 * episode_lenght
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
        min_by_day = 1640 * episode_lenght
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
        state = np.append(np.append(self.state, 0),0)
        
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
        print('Trade State: {0}'.format(self.trade))

    def _get_profit(self):
        profit = 0
        state = self.episode_data[self.nstep:(self.nstep+self.period)]
        price = state[-1][DFCLOSE]
        if self.trade and self.buy_price != None:
            # profit = price - (self.buy_price + self.spread)
            profit = price - self.buy_price
        if self.trade and self.sell_price != None:
            # profit = (self.sell_price - self.spread) - price
            profit = self.sell_price - price
        return profit

    def _take_profit(self):
        profit = 0
        
        price = self.state[-1][DFCLOSE]
        if self.trade and self.buy_price != None:
            profit = price - (self.buy_price + self.spread)
            # profit = price - self.buy_price
        if self.trade and self.sell_price != None:
            profit = (self.sell_price - self.spread) - price
            # profit = self.sell_price - price
        return profit
    
    def _check_stoploss(self):
        profit = 0
        state = self.episode_data[self.nstep:(self.nstep+self.period)]
        price = state[-1][DFLOW]

        if self.trade and self.buy_price != None:
            profit = price - (self.buy_price + self.spread)
            # profit = price - self.buy_price
        if self.trade and self.sell_price != None:
            profit = (self.sell_price - self.spread) - price
            # profit = self.sell_price - price
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

        if self.trade == False and action == BUY:
            self.trade = True
            self.total_trade += 1
            self.buy_price = price
            if self._get_profit() > 0:
                reward = 10
            elif self._get_profit() < 0:
                reward = -100
            else:
                reward = 0

        if self.trade == False and action == SELL:
            self.trade = True
            self.total_trade += 1
            self.sell_price = price
            if self._get_profit() > 0:
                reward = 10
            elif self._get_profit() < 0:
                reward = -100
            else:
                reward = 0

        if self.trade == True and (action == BUY or action == SELL):
            reward = -1

        if action == HOLD and self.trade == True:
            if self._get_profit() > 0:
                reward = 1
            elif self._get_profit() < 0:
                reward = -10
            else:
                reward = 0
        elif action == HOLD and self.trade == False:
            if self._get_profit() > 0:
                reward = 1
            elif self._get_profit() < 0:
                reward = -10
            else:
                reward = 0
        
        if self.trade == True and action == CLOSE:
            self.sold = (self._take_profit() * self.nlot) + self.sold
            self.trade_sold = self.sold
            
            # print("action CLOSE", (self._take_profit() * self.nlot))
            if self._take_profit() > 0:
                reward = 1000
            elif self._take_profit() < 0:
                reward = 10
            else:
                reward = self._take_profit() * self.nlot
            self.buy_price = None
            self.sell_price = None
            self.trade = False
        
        if self.trade == False and action == CLOSE:
            reward = -10

        if self.trade == True and self._take_profit() * self.nlot < -3 and (action == BUY or action == HOLD or action == SELL):
            reward = -1000


        if self.trade_sold < self.min_sold or self.sold < self.min_sold:
            done = True
            self.sold = (self._take_profit() * self.nlot) + self.sold
            self.trade_sold = self.sold
            reward = -10000

        # TEEEEESSSSSTTTTT
        if self.trade == True and self._check_stoploss() * self.nlot < -2:
            reward = -10000
            if self._take_profit() * self.nlot < 0:
                self.sold = self.sold - 4
                # print("action STOP LOSS", -0.1)
            elif self._take_profit() * self.nlot > 0:
                self.sold = self.sold - 4
                # print("action STOP LOSS", -0.1)
            self.trade_sold = self.sold
            
            self.buy_price = None
            self.sell_price = None
            self.trade = False

        # Update state
        self.state = self.episode_data[self.nstep:(self.nstep+self.period)]

        # Update trade sold
        self.trade_sold = (self._take_profit() * self.nlot) + self.sold

        trade_state = 0
        if (self.trade == True and self.buy_price != None):
            trade_state = 1
            state = np.append(np.append(self.state, trade_state),self._take_profit())
        if (self.trade == True and self.sell_price != None):
            trade_state = 2
            state = np.append(np.append(self.state, trade_state),self._take_profit())
        else:
            state = np.append(np.append(self.state, trade_state),self._take_profit())

        return state, reward, done