import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt


def plot_n_leaves(features, label, n_leaves):
    num_trees = []
    errors = []

    for num in range(1, n_leaves):
        num_trees.append(num)

    for n in range(1, n_leaves):
        X = np.array(features)
        y = np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, random_state=0)
        regressor = RandomForestRegressor(n_estimators=20, random_state=0, min_samples_leaf=n)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        mse = metrics.mean_squared_error(y_test, y_pred)
        errors.append(mse)

    fig, axs = plt.subplots(1)
    axs.set_title('Random Forests With Different Minimum Leaf Sizes')
    axs.plot(num_trees, errors)
    axs.set_xlabel('Minimum Samples for Leaves in Decision Tree (20 trees per forest)')
    axs.set_ylabel('MSE')

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('raw_csvs/bitcoin_truncated.csv', low_memory=False)
    df_label = df['market_price']
    df.drop(['market_price'], 1, inplace=True)
    df.drop(['date'], 1, inplace=True)
    df.drop(df.columns[0:2], 1, inplace=True)
    plot_n_leaves(df, df_label, 100)
