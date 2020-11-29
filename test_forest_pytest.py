import pickle
from random_forest_regressor import RandomForestRegressor, RandomTree
# tests the RandomForestRegressor class

points = [{'Date': '2016-07-17', 'Confirm Time': 7.76, 'Block Size': 0.68, 'Cost/TXN': 7.32, 'Difficulty': 213398925331.0, 'TXN Vol': 129816925.1, 'Hash Rate': 1665474.54, 'Market Cap': 10453324633.1, 'Miners Rev': 1302095.02, 'TXN/Block': 1160.45, 'Number of TXN': 182192.0, 'Unique Addresses': 356533.0, 'Total Bitcoin': 15763137.5, 'TXN Fees': 50.67, 'Trade Vol': 13030366.55, 'TXN/Trade Ratio': 60.58214548260001, 'Price': 679.051, 'Price Binary': 1},
          {'Date': '2016-07-16', 'Confirm Time': 8.98, 'Block Size': 0.79, 'Cost/TXN': 5.38, 'Difficulty': 213398925331.0, 'TXN Vol': 187503475.3, 'Hash Rate': 1432095.94, 'Market Cap': 10514437454.2, 'Miners Rev': 1127082.34, 'TXN/Block': 1600.59, 'Number of TXN': 216081.0, 'Unique Addresses': 367739.0, 'Total Bitcoin': 15761175.0, 'TXN Fees': 55.22, 'Trade Vol': 26928546.83, 'TXN/Trade Ratio': 93.1540589114, 'Price': 663.541, 'Price Binary': 0},
          {'Date': '2016-07-15', 'Confirm Time': 9.23, 'Block Size': 0.79, 'Cost/TXN': 5.49, 'Difficulty': 213398925331.0, 'TXN Vol': 164674255.32, 'Hash Rate': 1506352.77, 'Market Cap': 10423167437.6, 'Miners Rev': 1175290.03, 'TXN/Block': 1553.9, 'Number of TXN': 220655.0, 'Unique Addresses': 367953.0, 'Total Bitcoin': 15759487.5, 'TXN Fees': 58.38, 'Trade Vol': 34318017.31, 'TXN/Trade Ratio': 76.3547780853, 'Price': 664.8760000000001, 'Price Binary': 1},
          {'Date': '2016-07-14', 'Confirm Time': 10.73, 'Block Size': 0.81, 'Cost/TXN': 5.06, 'Difficulty': 213398925331.0, 'TXN Vol': 187860897.96, 'Hash Rate': 1379055.35, 'Market Cap': 10366219391.2, 'Miners Rev': 1070321.94, 'TXN/Block': 1677.63, 'Number of TXN': 218093.0, 'Unique Addresses': 367252.0, 'Total Bitcoin': 15757725.0, 'TXN Fees': 53.55, 'Trade Vol': 53026862.77, 'TXN/Trade Ratio': 86.47388884360001, 'Price': 659.6360000000001, 'Price Binary': 1},
          {'Date': '2016-07-13', 'Confirm Time': 9.26, 'Block Size': 0.77, 'Cost/TXN': 5.69, 'Difficulty': 213398925331.0, 'TXN Vol': 193691457.42, 'Hash Rate': 1516960.89, 'Market Cap': 10682486713.9, 'Miners Rev': 1213263.1, 'TXN/Block': 1531.7, 'Number of TXN': 219034.0, 'Unique Addresses': 379937.0, 'Total Bitcoin': 15756112.5, 'TXN Fees': 51.17, 'Trade Vol': 48343367.13, 'TXN/Trade Ratio': 39.4733175548, 'Price': 653.934, 'Price Binary': 0},
          {'Date': '2016-07-12', 'Confirm Time': 7.96, 'Block Size': 0.79, 'Cost/TXN': 5.49, 'Difficulty': 213398925331.0, 'TXN Vol': 166782085.8, 'Hash Rate': 1516960.89, 'Market Cap': 10240153706.8, 'Miners Rev': 1162507.11, 'TXN/Block': 1525.09, 'Number of TXN': 218089.0, 'Unique Addresses': 395742.0, 'Total Bitcoin': 15754325.0, 'TXN Fees': 57.33, 'Trade Vol': 26402272.48, 'TXN/Trade Ratio': 57.0098951949, 'Price': 664.8389999999999, 'Price Binary': 1}]

labels = [num for num in range(len(points))]
RFR = RandomForestRegressor(10, pool_num_attributes=4)
attributes = points[0].keys()

Tree = RandomTree()

def test_bootstrap():
    bootstrap_points, bootstrap_labels = RFR.bootstrap(points, labels, 4)
    print(bootstrap_points, bootstrap_labels)
    assert len(bootstrap_points) == len(bootstrap_labels) == 4

def test_possible_attr_vals():
    poss_vals = RFR.possible_attr_vals(points, attributes)
    for attr in attributes:
        for point in points:
            if attr in point:
                assert point[attr] in poss_vals[attr]

def test_build_forest():
    RFR.build_forest(points, labels)
    assert RFR.all_trained
