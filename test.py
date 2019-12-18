import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso

bike = pd.read_csv('/Users/user/Downloads/C06_learning_data/gongguan_best.csv')

x = bike.drop(['lent'], axis = 1)
y = bike['lent']

lm = LinearRegression()
print(cross_val_score(lm, x, y, cv = 4).mean())
lm_lasso = Lasso(alpha = 0.001, max_iter = 1000000)
print(cross_val_score(lm_lasso, x, y, cv = 4).mean())
lm_lasso = Lasso(alpha = 0.005, max_iter = 1000000)
print(cross_val_score(lm_lasso, x, y, cv = 4).mean())
lm_lasso = Lasso(alpha = 0.01, max_iter = 1000000)
print(cross_val_score(lm_lasso, x, y, cv = 4).mean())