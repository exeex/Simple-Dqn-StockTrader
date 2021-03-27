
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

        self.dates, self.stock_price_data, self.train_data = self._build_dataset(
            "TSM")  # 台積電美股代號

        self.window_size = 30  # 根據過去多少天的資料做決策
        column = self.train_data.shape[0]  # 觀察的欄位數目
        self.n_features = self.window_size * column  # model input vector大小

        self.capital = 10000.0  # 本金(USD)
        self.stock_num = 0  # 持有股數 (若為負值 則為賣空股數)

        self._account_balance = 0.0  # 帳戶結餘
        self._select_idx = 0

    def _build_dataset(self, stock_name):

        dataset = data_reader.DataReader(stock_name, data_source="yahoo")

        # build original dataset
        dates = [str(date) for date in dataset.index]
        close = dataset['Close'].to_numpy()  # 休市價格
        stock_price_data = np.stack([close], axis=0)  # shape = [1, Date]

        # build transformed dataset (for model training) 
        """
        說明:
        你可以在此處自行設計策略!!!
        只要確保train_data 輸出shape為 [欄位書目, 資料筆數] 即可

        習題: 增加周線/月線/季線的資料欄位
        Hint: 對休市價格(Close) 做移動平均濾波器(Moving Average Filter)
        """
        close = dataset['Close'].to_numpy()  # 休市價格
        volume = dataset['Volume'].to_numpy()  # 交易量

        # global normalize
        norm_close = close / close.max()
        norm_volume = volume / volume.max()

        train_data = np.stack([norm_close, norm_volume],
                              axis=0)  # shape = [欄位書目, 資料筆數]

        return dates, stock_price_data, train_data

    def get_observations(self, idx):
        # 注意idx會不會存取越界!
        return self.train_data[:, idx-self.window_size: idx].flatten()

    def get_date(self):
        return str(self.dates[self._select_idx]).split()[0]

    def reset(self):

        # 設定初始資金
        self._account_balance = self.capital

        # 隨機選擇某日開始操作股票
        self.stock_num = 0
        self._select_idx = random.randint(
            self.window_size, self.stock_price_data.shape[1]-2)  # 注意idx會不會存取越界!

        print(f"以{self.get_date()}為初始日期開始模擬操盤..")

        # return observation
        return self.get_observations(self._select_idx)

    def step(self, action_id):

        # 處理action
        stock_price = self.stock_price_data[0, self._select_idx]

        if action_id == 0:  # buy
            self._account_balance -= stock_price
            self.stock_num += 1
        elif action_id == 1:  # hodl
            pass
        elif action_id == 2:  # sell
            self._account_balance += stock_price
            self.stock_num -= 1

        # 前進一天
        self._select_idx += 1

        # 計算新一天的 observation
        s_ = self.get_observations(self._select_idx)

        # 計算Reward
        account_value = self.stock_num * stock_price + self._account_balance  # 計算當前帳戶淨值
        profit = account_value / self.capital - 1  # 當前利潤率

        print(
            f"{self.get_date()} : {self.action_space[int(action_id)]}, profit= {profit}")

        if profit <= -0.2:  # 利潤小於-20%止損出場
            reward = -1.0
            done = True
        else:
            reward = profit
            done = False

        if self._select_idx+1 >= self.stock_price_data.shape[1]:  # 若日期超出範圍停止模擬
            done = True

        return s_, reward, done
