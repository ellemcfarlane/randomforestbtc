import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

#dataset = pd.read_csv('petrol_consumption.csv')
# print(dataset.head())

class RandomTreeBTC:
    def __init__(self, x):
        self.x = 10

def load_data_frame(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

# divide data into points and labels
dataset = load_data_frame('data_frames/petrol_df.pkl')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# split into training vs test attributes/labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# scale values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# evaluate algo peformance
print('Mean Abs Error', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Sq Error', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
