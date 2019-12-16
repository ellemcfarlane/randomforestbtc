# Random Forest BTC
Uses a random forest (RF) regression model to predict bitcoin prices 3 days ahead.
Trained on data from 2010-2019.
## Features
* Prediction of:
  * BTC price in USD 3 days ahead of input BTC data
* Visualization of:
  * RF model fit to test and training data

## Basic Usage
# Requirements

# Current predictions
Run the Main.py file in the front folder; click the http://127.0.0.1:5000/ link.
The resulting prediction is based on the current day's BTC data. 

# Retrain model
To train your own model, and view prediction results, do the following:
```
    RFregressor = RandomForestRegressor(20)
    RFregressor.build_forest(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(y_pred)
```
where 20 in this case represents number of trees in the forest, X_train and X_test lists of dictionaries, each data point as a dictionary
and y_pred a list of numbers, representing predictions of the tree.

# Results of our model
Without retraining our model, to see the results based on 20 trees, bootstrap sample of original
training size, and considering all features, simply run the random_forest_regressor.py file in the main folder.

These results should match those shown in the performance.txt file
## Tests
Run the test_forest_pytest.py file.

## Acknowledgements

## Authors
Ryan Shuey, Evan Truong, & Elle McFarlane

