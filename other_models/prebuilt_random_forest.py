import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

#############################################################################

# Old BTC

# divide data into points and labels
# dataset = pd.read_csv('../raw_csvs/df_final.csv', low_memory=True)
# dataset['Price'] = dataset['Price'].shift(3)
# dataset.drop(0, axis=0, inplace=True)
# dataset.drop(1, axis=0, inplace=True)
# dataset.drop(2, axis=0, inplace=True)
# y_orig = dataset.loc[:, 'Price'].values[::-1]
# X_orig = dataset.drop(columns=['Unnamed: 0', 'Date', 'Price']).reindex(index=dataset.index[::-1]).to_numpy()
# dataset = dataset.sample(frac=1, random_state=0)
# y = dataset.loc[:, 'Price'].values
# print(dataset.head())
# dataset.drop(columns=['Unnamed: 0', 'Date', 'Price'], inplace=True)

#############################################################################

# New BTC

dataset = pd.read_csv('../raw_csvs/bitcoin_truncated.csv', low_memory=True)
dataset['market_price'] = dataset['market_price'].shift(-1)
dataset.drop(len(dataset) - 1, axis=0, inplace=True)
y_orig = dataset.loc[:, 'market_price'].values
X_orig = dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'date', 'market_price']).to_numpy()
dataset = dataset.sample(frac=1, random_state=0)
y = dataset.loc[:, 'market_price'].values
dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'date', 'market_price'], inplace=True)
print("dataset preview:")
print(dataset.head())


X = dataset.to_numpy()
# split into training vs test attributes/labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_train_pred = regressor.predict(X_train)
y_full = regressor.predict(X_orig)

print("y-pred", y_pred[-10:])


# with open('old_btc_forest.pkl', 'wb') as file:
#     pickle.dump(regressor, file)


# Evaluate algo performance
errors = abs(y_pred - y_test)
map = 100 * np.mean(errors / y_test)
accuracy = 100 - map

print('Performance')
print('Accuracy = {:0.2f}%.'.format(accuracy))
print('Mean Abs Error', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Sq Error', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Visualize results
x_test = range(len(y_pred))
x_train = range(len(y_train_pred))
x_full = range(len(y_pred) + len(y_train_pred))

fig, axs = plt.subplots(3)
axs[0].set_title('Bitcoin Price RF Predictions')
axs[0].scatter(x_train, y_train, c='black')
axs[0].scatter(x_train, y_train_pred, c='red', marker=".")

axs[1].scatter(x_test, y_test, c='black')
axs[1].scatter(x_test, y_pred, c='red', marker=".")

axs[2].scatter(x_full, y_orig, c='black')
axs[2].plot(x_full, y_full, c='red')
axs[2].set_xlabel('Relative dates from 2012-2016')

plt.show()
