# Import the libraries
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import the training set
dataset_train = pd.read_csv('/Users/bill/Downloads/TSM-Train.csv')  # 讀取訓練集
training_set = dataset_train.iloc[:, 5:6].values  # 取「Adj」欄位值

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)  # normalization in [0,1]

X_train = []   # 預測點的前 60 天的資料
y_train = []   # 預測點
for i in range(60, 1240):  # 1240 是訓練集總數
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)  # 轉成numpy array的格式，以利輸入 RNN

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# reshape to 3-dimension: [stock prices, timesteps, indicators]
# LSTM 所要求的三維格式 [樣本,輸入的單位(ex一個單詞),輸入那個單位所觀察到的特徵]

# print(X_train)

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
import keras


# Prediction
dataset_test = pd.read_csv('/Users/bill/Downloads/TSM-Test.csv')
real_stock_price = dataset_test.iloc[:, 5:6].values

dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis=0)
# combine vertically
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs) # Feature Scaling

X_test = []
for i in range(60, 326):  # timesteps一樣60； 80 = 先前的60天資料+2019年的266天資料
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension

decaylist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def LSTMmodel(drop,dec):
    global X_test, real_stock_price

    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(drop))
    # unit - 神經元數目 / return_sequences = True 返回全部序列 / add - 添加神經層
    # Dropout(在 0 和 1 之间的浮点数)-单元的丢弃比例，用于输入的线性转换 (to avoid overfit)


    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(drop))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(drop))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(drop))
    # 由於這邊的第四層 LSTM Layer 即將跟 Ouput Layer 做連接，因此注意這邊的
    # return_sequences 預設值為 False （也就是不用寫上 return_sequences）

    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Dense - 全連結神經層(回歸到此曾做分類,通常為一維)

    # Compiling 激活模型
    adam = keras.optimizers.Adam(decay=dec)
    regressor.compile(optimizer = adam, loss = 'mean_squared_error')
    # optimizer(優化器選擇) /loss(定義誤差函數)

    # 進行訓練
    regressor.fit(X_train, y_train, epochs=100, batch_size=32)
    # epochs 訓練次數  / batch_size 梯度下降的樣本數
    a = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(a)  # to get the original scale
    h = []
    for i in range(len(real_stock_price)):
        h.append((float(predicted_stock_price[i]) - float(real_stock_price[i])) ** 2)
    return (sum(h) / 1240), predicted_stock_price


# 優化模型
drop_MSE = {}
for parameter_drop in dlist:
    value1, no_use1 = LSTMmodel(parameter_drop, 0.2)
    drop_MSE[parameter_drop] = value1

lists1 = sorted(drop_MSE.items())
x1, y1 = zip(*lists1)
plt.figure(figsize=(15,15))
plt.subplot(221)
plt.plot(x1, y1)
plt.title('Finding the best parameter_drop')
plt.xlabel('Dropout')
plt.ylabel('Mean Square Error')


decay_MSE = {}
for parameter_decay in decaylist:
    value2, no_use2 = LSTMmodel(min(drop_MSE), parameter_decay)
    decay_MSE[parameter_decay] = value2
lists2 = sorted(decay_MSE.items())
x2, y2 = zip(*lists2)
plt.subplot(222)
plt.plot(x2, y2)
plt.title('Finding the best parameter_decay')
plt.xlabel('Dropout')
plt.ylabel('Mean Square Error')


no_use3, predicted_price = LSTMmodel(min(drop_MSE), min(decay_MSE))

print(predicted_price)

import csv

# 開啟輸出的 CSV 檔案
with open('/Users/bill/Desktop/New predict .csv', 'w', newline='') as csvFile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvFile)
    writer.writerow(predicted_price)


# Visualising the results
plt.subplot(212)
plt.plot(real_stock_price, color = 'red', label = 'Real TSM Stock Price')  # 紅線表示真實股價
plt.plot(predicted_price, color = 'blue', label = 'Predicted TSM Stock Price')  # 藍線表示預測股價
plt.title('TSM Price Prediction')
plt.xlabel('Time')
plt.ylabel('TSM Stock Price')
plt.legend()

plt.show()