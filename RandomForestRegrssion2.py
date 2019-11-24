import pandas as pd
import random
from sklearn import metrics
import math


def df_to_dict(df):
    points = df.to_dict('records')
    return points


def process_data(points, label):
    labels = []
    for p in points.copy():
        labels.append(p[label])
        del p[label]
    return points, labels


################################


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

<<<<<<< HEAD
    def classify(self, new_point):
        if self.attribute is None:
            return self.value
        if new_point[self.attribute[0]] <= self.attribute[1]:
            return self.true_child.classify(new_point)
        else:
            return self.false_child.classify(new_point)
=======

if __name__ == '__main__':
    # df = pd.read_csv('raw_csvs/petrol_consumption.csv', low_memory=False)
    df = pd.read_csv('raw_csvs/df_final.csv', parse_dates=['Date'], low_memory=False)
    df.drop(['Date'], 1, inplace=True)

    p = get_labels(read_df(df), 'Price')[0]
    l = get_labels(read_df(df), 'Price')[1]

    n = Node(p, l)
    n.build_tree()
    leaf_values(n)











>>>>>>> 4428448d30c6b2cc24bb9bafa7c5928873f59378


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
        for i in range(n_trees):
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
        return sum(tree_vals)/len(tree_vals)


if __name__ == '__main__':
    df = pd.read_csv('raw_csvs/petrol_consumption.csv', low_memory=False)
    read = df_to_dict(df)
    proc = process_data(read, 'Petrol_Consumption')
    points = proc[0]
    labels = proc[1]

    # n = Node(points, labels)
    # n.build_tree()
    # leaf_values(n)

    test_data = {'Petrol_tax': 9,
                 'Average_income': 3500,
                 'Paved_Highways': 2000,
                 'Population_Driver_licence(%)': .53}

    rf = RandomForestRegressor(points, labels, 30, 0.3)
    print(rf.predict(test_data))
