import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


def plot_n_trees(features, label, n_trees):
    num_trees = []
    errors = []

    for num in range(1, n_trees):
        num_trees.append(num)

    for n in range(1, n_trees):
        X = np.array(features)
        y = np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, random_state=0)
        regressor = RandomForestRegressor(n_estimators=n, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        errors.append(mse)

    errors2 = []

    for n in range(1, n_trees):
        X = np.array(features)
        y = np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False, random_state=0)
        regressor = RandomForestRegressor(n_estimators=n, random_state=0)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        errors2.append(mse)

    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=.6)
    axs[0].set_title('Random Forests With Shuffled Dataset')
    axs[0].plot(num_trees, errors)
    axs[0].set_xlabel('Number of Estimators')
    axs[0].set_ylabel('MSE')

    axs[1].set_title('Random Forest Without Shuffled Dataset')
    axs[1].plot(num_trees, errors2)
    axs[1].set_xlabel('Number of Estimators')
    axs[1].set_ylabel('MSE')

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('raw_csvs/bitcoin_final.csv', low_memory=False)
    df_label = df['market_price']
    df.drop(['market_price'], 1, inplace=True)
    df.drop(['date'], 1, inplace=True)
    plot_n_trees(df, df_label, 200)