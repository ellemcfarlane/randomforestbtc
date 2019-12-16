import pandas as pd
import random
import numpy as np
import math
from sklearn import metrics


def df_to_dict(df):
    points = df.to_dict('records')
    return points


def split_by_label(points, label):
    labels = []
    for p in points.copy():
        labels.append(p[label])
        del p[label]
    return points, labels


def split_by_value(points, labels, split_feature, split_value):
    true_points = []
    true_labels = []
    false_points = []
    false_labels = []

    vals = []
    for i in range(len(points)):
        vals.append(points[i][split_feature])
    max_val = max(vals)

    for i in range(len(points)):
        if points[i][split_feature] == max_val and split_value == max_val:
            false_points.append(points[i])
            false_labels.append(labels[i])
        elif points[i][split_feature] <= split_value:
            true_points.append(points[i])
            true_labels.append(labels[i])
        else:
            false_points.append(points[i])
            false_labels.append(labels[i])

    return true_points, true_labels, false_points, false_labels


def best_split(points, labels):
    scores = []
    min_score = float('inf')
    for i in range(len(points)):
        for key, val in points[i].items():
            split = split_by_value(points, labels, key, val)
            if len(split[1]) == 0:
                continue
            if len(split[3]) == 0:
                continue
            true_mean = sum(split[1])/len(split[1])
            false_mean = sum(split[3])/len(split[3])
            true_mean_arr = []
            false_mean_arr = []
            for i in range(len(split[1])):
                true_mean_arr.append(true_mean)
            for i in range(len(split[3])):
                false_mean_arr.append(false_mean)
            true_rmse = math.sqrt(metrics.mean_squared_error(split[1], true_mean_arr))
            false_rmse = math.sqrt(metrics.mean_squared_error(split[3], false_mean_arr))
            true_score = true_rmse * len(split[1])
            false_score = false_rmse * len(split[3])
            score = true_score + false_score
            if score < min_score:
                min_score = score
            scores.append((key, val, score))

    for split in scores:
        if split[2] == min_score:
            return split


def leaf_values(node):
    if node.attribute is None:
        print(node.points)
        print(node.labels)
        print()
    else:
        leaf_values(node.false_child)
        leaf_values(node.true_child)


class Node:
    def __init__(self, points, labels):
        self.points = points
        self.labels = labels
        self.attribute = None
        self.length = len(points)
        self.value = sum(labels)/len(labels)

        value_arr = []
        for i in range(self.length):
            value_arr.append(self.value)

        self.rmse = math.sqrt(metrics.mean_squared_error(labels, value_arr))
        self.false_child = None
        self.true_child = None

    def build_tree(self):
        if self.length <= 1:
            return
        best_split_val = best_split(self.points, self.labels)
        if best_split_val is None:
            return
        split = split_by_value(self.points, self.labels, best_split_val[0], best_split_val[1])

        self.attribute = best_split_val[0], best_split_val[1]
        self.false_child = Node(split[2], split[3])
        self.true_child = Node(split[0], split[1])
        self.false_child.build_tree()
        self.true_child.build_tree()

    def classify(self, new_point):
        if self.attribute is None:
            return self.value
        if new_point[self.attribute[0]] <= self.attribute[1]:
            return self.true_child.classify(new_point)
        else:
            return self.false_child.classify(new_point)


def subset_dataset(points, labels, n):
    sub_points = []
    sub_labels = []
    for i in range(n):
        r = random.randint(0, n)
        sub_points.append(points[r])
        sub_labels.append(labels[r])
    return sub_points, sub_labels


class RandomForestRegressor:
    def __init__(self, points, labels, n_trees, subset_portion):
        self.forest = []
        self.tree_values = None
        for i in range(n_trees):
            print(i)
            n = int(len(points) * subset_portion)
            subset_data = subset_dataset(points, labels, n)
            sub_points = subset_data[0]
            sub_labels = subset_data[1]
            node = Node(sub_points, sub_labels)
            node.build_tree()
            self.forest.append(node)

    def predict(self, new_data):
        tree_vals = []
        for tree in self.forest:
            tree_vals.append(tree.classify(new_data))
        self.tree_values = tree_vals
        return sum(tree_vals)/len(tree_vals)

    def predict_test_set(self, x_test):
        predictions = []
        for point in x_test:
            pred = self.predict(point)
            predictions.append(pred)
        return predictions


if __name__ == '__main__':
    # df = pd.read_csv('raw_csvs/petrol_consumption.csv', low_memory=False)
    # read = df_to_dict(df)
    # train = read[:38]
    # test = read[38:]
    #
    # train_split = split_by_label(train, 'Petrol_Consumption')
    # x_train = train_split[0]
    # y_train = train_split[1]
    #
    # rf = RandomForestRegressor(x_train, y_train, 200, .8)
    # test_split = split_by_label(test, 'Petrol_Consumption')
    # x_test = test_split[0]
    # y_test = test_split[1]
    # predictions = rf.predict_test_set(x_test)
    # print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#############################################################################

    # old data

    df = pd.read_csv('raw_csvs/df_final.csv', low_memory=True)
    df.drop(['Date'], 1, inplace=True)
    read = df_to_dict(df)

    training_size = 950
    train = read[:training_size]
    test = read[training_size:]

    train_split = split_by_label(train, 'Price')
    x_train = train_split[0]
    y_train = train_split[1]

    rf = RandomForestRegressor(x_train, y_train, 20, .99)
    test_split = split_by_label(test, 'Price')
    x_test = test_split[0]
    y_test = test_split[1]
    predictions = rf.predict_test_set(x_test)
    print(predictions[:-10])
    print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#############################################################################

    # new data

    # df = pd.read_csv('raw_csvs/bitcoin_final.csv', low_memory=True)
    # df.drop(['date'], 1, inplace=True)
    # read = df_to_dict(df)
    #
    # training_size = 950
    # train = read[:training_size]
    # test = read[training_size:]
    #
    # train_split = split_by_label(train, 'market_price')
    # x_train = train_split[0]
    # y_train = train_split[1]
    #
    # rf = RandomForestRegressor(x_train, y_train, 20, .99)
    # test_split = split_by_label(test, 'market_price')
    # x_test = test_split[0]
    # y_test = test_split[1]
    # predictions = rf.predict_test_set(x_test)
    # print(predictions[:-10])
    # print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
