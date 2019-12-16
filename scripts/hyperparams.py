from pprint import pprint
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

def load_data_frame(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 2, stop = 100, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    #pprint(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=400, cv=3, verbose=2,
                                   random_state=0, n_jobs=-1)
    # dataset = pd.read_csv('raw_csvs/df_final.csv', low_memory=True)
    # dataset['Price'] = dataset['Price'].shift(3)
    # dataset.drop(0, axis=0, inplace=True)
    # dataset.drop(1, axis=0, inplace=True)
    # dataset.drop(2, axis=0, inplace=True)
    # dataset = dataset.sample(frac=1, random_state=0)
    # y = dataset.loc[:, 'Price'].values
    # print(dataset.head())
    # dataset.drop(columns=['Unnamed: 0', 'Date', 'Price'], inplace=True)
    # X = dataset.to_numpy()
    # # split into training vs test attributes/labels
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)

    dataset = pd.read_csv('raw_csvs/bitcoin_truncated.csv', low_memory=True)
    dataset['market_price'] = dataset['market_price'].shift(-1)
    dataset.drop(len(dataset) - 1, axis=0, inplace=True)
    y_orig = dataset.loc[:, 'market_price'].values
    # X_orig = dataset.drop(columns=['market_price', 'date', 'Unnamed: 0']).to_numpy()
    X_orig = dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'date', 'market_price']).to_numpy()
    print(dataset.head())
    dataset = dataset.sample(frac=1, random_state=0)
    y = dataset.loc[:, 'market_price'].values
    # dataset.drop(columns=['market_price', 'date', 'Unnamed: 0'], inplace=True)
    dataset.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'date', 'market_price'], inplace=True)
    print(dataset.head())

    X = dataset.to_numpy()
    # split into training vs test attributes/labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)

    # Fit the random search model
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)


    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        map = 100 * np.mean(errors / test_labels)
        accuracy = 100 - map
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy


    base_model = RandomForestRegressor(n_estimators=20, random_state=0)
    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_train, y_train)

    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_train, y_train)

    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))