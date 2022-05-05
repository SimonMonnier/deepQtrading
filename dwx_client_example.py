
import json
from time import sleep
from threading import Thread
from os.path import join, exists
from traceback import print_exc
from random import random
from datetime import datetime, timedelta

from api.dwx_client import dwx_client

import numpy as np

from src.agent import DQNAgent, Config


DFOPEN = 0
DFHIGH = 1
DFLOW = 2
DFCLOSE = 3

BUY = 0
SELL = 1
CLOSE = 2
HOLD = 3

"""

Example dwxconnect client in python


This example client will subscribe to tick data and bar data. It will also request historic data. 
if open_test_trades=True, it will also open trades. Please only run this on a demo account. 

"""

class tick_processor():

    def __init__(self, MT4_directory_path, 
                 sleep_delay=0.005,             # 5 ms for time.sleep()
                 max_retry_command_seconds=10,  # retry to send the commend for 10 seconds if not successful. 
                 verbose=True
                 ):

        
        self.config = Config(target_update=10, 
                lr=0.001,
                lr_min=0.001, 
                lr_decay=2000, 
                gamma=0.50,
                loss='huber', 
                memory_size=161800, 
                batch_size=30, 
                eps_start=0.01,
                eps_min=0.01, 
                eps_decay=1500)

        self.agent = DQNAgent(config=self.config, id="1618")
        self.agent.load(r"C:/Users/smonn/Desktop/DeepTrading/src/models/1618-1651689609.pt")

        # if true, it will randomly try to open and close orders every few seconds. 
        self.open_test_trades = False
        self.trade = 0
        self.last_open_time = datetime.utcnow()
        self.last_modification_time = datetime.utcnow()

        self.dwx = dwx_client(self, MT4_directory_path, sleep_delay, 
                              max_retry_command_seconds, verbose=verbose)
        
        self.dwx.start()
        
        # account information is stored in self.dwx.account_info.
        # print("Account info:", self.dwx.account_info)

        # subscribe to tick data:
        self.dwx.subscribe_symbols(['EURUSD'])
        self.res = [[0,0,0,0]]
        self.buy_price = None
        self.sell_price = None
        self.spread = 0.0002
        
    def _take_profit(self, ask, bid):
        profit = 0
        self.spread = ask - bid
        price = ask
        if self.trade and self.buy_price != None:
            price = ask
            profit = price - (self.buy_price + self.spread)
            
        if self.trade and self.sell_price != None:
            price = bid
            profit = (self.sell_price - self.spread) - price
            
        return profit

    def on_tick(self, symbol, bid, ask):
        time = self.dwx._last_market_data.get(symbol)
        if time != None:
            time = time['time']
            time = time.split('.', 3)
            year = time[0]
            month = time[1]
            day = time[2].split()
            hour = day[1].split(':')
            minute = hour[1]
            hour = hour[0]
            day = day[0]
            
            time = datetime(int(year), int(month), int(day), int(hour), int(minute))
            
            end = time + timedelta(hours=1)
            start = end - timedelta(hours=300)  # last 30 days
            self.dwx.get_historic_data(symbol, 'H1', start.timestamp(), end.timestamp())
            sleep(1)
            datas = self.dwx.historic_data[symbol + '_H1']
            
            res = []
            for key in datas.keys():
                row = [datas[key]['open'], datas[key]['high'], datas[key]['low'], datas[key]['close']]
                res.append(row)
            res = res[len(res) - 200 :len(res) + 1]
            
            if (self.res[0][0] != res[0][0]):
                self.res = res

                if (self.dwx.open_orders == {}):
                    self.trade = 0
                    self.buy_price = None
                    self.sell_price = None
                    state = np.append(np.append(res, self.trade), 0)
                else:
                    self.trade = 1
                    state = np.append(np.append(res, self.trade),self._take_profit(ask,bid))

                action = self.agent._select_action(state,episode=1)
                
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                print(self._take_profit(ask,bid))
                if (action == BUY):
                    print("BBBBBBBBBBBBUUUUUYYYYYYYYYYYY")

                if (action == BUY and self.dwx.open_orders == {}):
                    order_type = 'buy'
                    price = ask
                    self.buy_price = ask
                    self.spread = ask - bid
                    self.dwx.open_order(symbol=symbol, order_type=order_type, stop_loss=price - self.spread * 2, magic=1618, price=price, lots=0.1)
                if (action == SELL):
                    print("SSSSSSSSSSSSEEEEEEEEEEEEEEEELLLLLLLL")
  
                if (action == SELL and self.dwx.open_orders == {}):
                    order_type = 'sell'
                    price = bid
                    self.sell_price = bid
                    self.spread = ask - bid
                    self.dwx.open_order(symbol=symbol, order_type=order_type, stop_loss=price + self.spread * 2, magic=1618, price=price, lots=0.1)
                if (action == CLOSE):
                    print("CCCCCCCCLLLLLLLLLLLLOOOOOOOOSSSSSSEEEEEEE")

                    self.dwx.close_all_orders()
                    self.buy_price = None
                    self.sell_price = None
                if (action == HOLD):
                    print("HHHHHOOOOOOOOOLLLLDDDDDDDD")
  

    def on_bar_data(self, symbol, time_frame, time, open_price, high, low, close_price, tick_volume):
        
        print('on_bar_data:', symbol, time_frame, datetime.utcnow(), time, open_price, high, low, close_price)

    
    def on_historic_data(self, symbol, time_frame, data):
        
        # you can also access the historic data via self.dwx.historic_data. 
        print('historic_data:', symbol, time_frame, f'{len(data)} bars')


    def on_historic_trades(self):

        print(f'historic_trades: {len(self.dwx.historic_trades)}')
    

    def on_message(self, message):

        if message['type'] == 'ERROR':
            print(message['type'], '|', message['error_type'], '|', message['description'])
        elif message['type'] == 'INFO':
            print(message['type'], '|', message['message'])


    # triggers when an order is added or removed, not when only modified. 
    def on_order_event(self):
        
        print(f'on_order_event. open_orders: {len(self.dwx.open_orders)} open orders')



# MT4_files_dir = 'C:/Users/smonn/AppData/Roaming/MetaQuotes/Terminal/AE2CC2E013FDE1E3CDF010AA51C60400/MQL5/Files'
MT4_files_dir = 'C:/Users/smonn/AppData/Roaming/MetaQuotes/Tester/AE2CC2E013FDE1E3CDF010AA51C60400/Agent-127.0.0.1-3000/MQL5/Files'
processor = tick_processor(MT4_files_dir)

while processor.dwx.ACTIVE:
    sleep(1)


