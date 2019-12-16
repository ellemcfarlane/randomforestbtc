import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == '__main__':
    # reads in csv file and drops the date column
    df = pd.read_csv('../raw_csvs/bitcoin_truncated.csv', low_memory=False)
    df['market_price'] = df['market_price'].shift(-1)
    df.drop(len(df)-1, axis=0, inplace=True)
    df.drop(['date'], 1, inplace=True)
    df = df.sample(frac=1, random_state=0)

    # makes a new data frame of the label
    df_label = df['market_price']
    df.drop(['market_price', 'Unnamed: 0', 'Unnamed: 0.1'], 1, inplace=True)

    # splits into training and testing data then fits the model
    X_train, X_test, y_train, y_test = train_test_split(df, df_label, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # prints the r squared coefficient of each feature
    coeff = pd.DataFrame(regressor.coef_, df.columns, columns=['Coefficient'])
    print(coeff)
    print()

    # predicts test data and prints error statistics
    y_pred = regressor.predict(X_test)

    print('Performance')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print()
    print(y_pred[-10:])