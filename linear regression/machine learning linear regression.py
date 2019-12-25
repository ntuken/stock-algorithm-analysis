import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
        
data = pd.read_csv("merged-train-data\\clear_merged-trained.csv")
# print(data.head())
# print(data.describe())
cp_name = data.iloc[:,4].unique()
print(cp_name)
aapl = data[data.company == "AAPL"]
DJ = data[data.company=='DOW-JONES']
SOXL = data[data.company=="SOXL"]
INTC = data[data.company=="INTC"]
TSMC = data[data.company=="TSMC"]
TSMC_correct_date = TSMC.Date.apply(lambda x:x.replace("/","-"))


TSMC["new_date"] = TSMC_correct_date
#print(TSMC["new_date"])


DJ_date = set(DJ["Date"])
SOXL_date = set(SOXL["Date"])
aapl_date = set(aapl["Date"])
INTC_date = set(INTC["Date"])
TSMC_date = set(TSMC["new_date"])

union_date = DJ_date&SOXL_date&aapl_date&INTC_date
union_date = union_date&TSMC_date
print(len(union_date))

DJ_index = DJ.Date.isin(union_date)
DJ = DJ.loc[DJ_index,"Adj.Close"]

SOXL_index = SOXL.Date.isin(union_date)
SOXL = SOXL.loc[SOXL_index,"Adj.Close"]

aapl_index = aapl.Date.isin(union_date)
aapl = aapl.loc[aapl_index,"Adj.Close"]

INTC_index = INTC.Date.isin(union_date)
INTC = INTC.loc[INTC_index,"Adj.Close"]

TSMC_index = TSMC.new_date.isin(union_date)
TSMC = TSMC.loc[TSMC_index,"Adj.Close"]

D = DJ.values.reshape(-1,1)
S = SOXL.values.reshape(-1,1)
a = aapl.values.reshape(-1,1)
I = INTC.values.reshape(-1,1)
T = TSMC.values.reshape(-1,1)
feature = np.hstack((D,S,a,I))
lm = LinearRegression()
lm.fit(feature,T)

print("coefficients:",lm.coef_)
print("Intercept: ",lm.intercept_)
print("R square : ",lm.score(feature,T))

TSMC_future = pd.read_csv("TSM (1).csv")
TSMC_future_correct_date = TSMC_future.Date.apply(lambda x:x.replace("/","-"))
TSMC_future["new_date"] = TSMC_future_correct_date

# pre_data = pre_data["new_date"]
# print(len(pre_data.loc[:,"Adj Close"]))


DJ_future = pd.read_csv("DOW-JONES.csv")

A_future = pd.read_csv("AAPL.csv")

S_future = pd.read_csv("SOXL.csv")

I_future = pd.read_csv("INTC.csv")


D = DJ_future.loc[0:212,"Adj Close"].values.reshape(-1,1)
A = A_future.loc[0:212,"Adj Close"].values.reshape(-1,1)
S = S_future.loc[0:212,"Adj Close"].values.reshape(-1,1)
I = I_future.loc[0:212,"Adj Close"].values.reshape(-1,1)
x_future = np.hstack((D,A,S,I))
TSMC_predict = lm.predict(x_future)

import matplotlib.pyplot as pyplot

# plt.scatter(predicted,y,s=2)
# plt.plot(predict_y, predict_y, 'ro')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
# plt.xlabel('Predicted')
# plt.ylabel('Measured')

pyplot.scatter(TSMC_predict,TSMC_future.loc[0:212,"Adj Close"],s=2)
pyplot.plot(TSMC_predict,TSMC_predict,"ro")
pyplot.show()
