
import numpy as np
import time
import sys
import random
import pandas_datareader as data_reader


class StockTradeAgent:
    def __init__(self):

        super(StockTradeAgent, self).__init__()
        self.action_space = ['buy', 'hodl', 'sell']  # 買1股, 不買不賣, 賣1股(或賣空1股)
        self.n_actions = len(self.action_space)

        self.window_size = 10  # 根據過去多少天的資料做決策
        column = 1  # 觀察的欄位數目
        self.n_features = self.window_size * column  # input vector大小
        self.dates, self.dataset = self._build_dataset("TSM")  # 台積電美股代號

        self.capital = 10000.0  # 本金(USD)
        self.stock_num = 0  # 持有股數 (若為負值 則為賣空股數)

        self._account_balance = 0.0  # 帳戶結餘
        self._select_idx = 0

    def _build_dataset(self, stock_name):
        # Complete the dataset loader function
        dataset = data_reader.DataReader(stock_name, data_source="yahoo")

        dates = [str(date) for date in dataset.index]
        close = dataset['Close'].to_numpy()  # 休市價格

        data = np.stack([close], axis=0)  # shape = [Column, Date]

        return dates, data

    def get_window_features(self, idx):
        return self.dataset[:, idx-self.window_size: idx].flatten()  # 注意idx會不會存取越界!

    def get_date(self):
        return str(self.dates[self._select_idx]).split()[0]

    def reset(self):

        # 設定初始資金
        self._account_balance = self.capital

        # 隨機選擇某日開始操作股票
        self._select_idx = random.randint(self.window_size, self.dataset.shape[1]-1)  # 注意idx會不會存取越界!

        print(f"以{self.get_date()}為初始日期開始模擬操盤..")

        # return observation
        return self.get_window_features(self._select_idx)

    def step(self, action_id):

        # 處理action
        stock_price = self.dataset[0, self._select_idx]

        if action_id == 0:  # buy
            self._account_balance -= stock_price
            self.stock_num +=1
        
        elif action_id == 1:  # hodl
            pass

        elif action_id == 2:  # sell
            self._account_balance += stock_price
            self.stock_num -=1

        # 前進一天
        self._select_idx += 1

            

        # 計算新一天的 observation
        s_ =self.get_window_features(self._select_idx)


        # 計算Reward

        account_value = self.stock_num * stock_price + self._account_balance #計算當前帳戶淨值
        profit = account_value / self.capital - 1 # 當前利潤率
        
        
        print(f"{self.get_date()} : {self.action_space[int(action_id)]}, profit= {profit}")

        if profit <= -0.2: # 利潤小於-20%止損出場
            reward = -1.0
            done = True 
        else:
            reward = profit
            done = False
        

        if self._select_idx+1 >= self.dataset.shape[1]:
            done = True

        return s_, reward, done

