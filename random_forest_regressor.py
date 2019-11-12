from random import randint
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# size of subsample for num of subsamp boots = sqrt of num training samps
class RandomForestRegressor:
    def __init__(self, num_estimators, pool_num_attributes=0):
        self.num_estimators = num_estimators
        self.pool_num_attributes = pool_num_attributes
        self.trees = [RandomTree() for num in num_estimators]
        self.all_trained = False

    def build_forest(self, points, labels):
        sample_size = np.sqrt(len(points))
        attributes = points[0].keys
        possible_vals = self.possible_attr_val(points, attributes)
        for tree in self.trees:
            bootstrap_points, bootstrap_labels = self.bootstrap(points, labels, sample_size)
            tree.train(bootstrap_points, bootstrap_labels, possible_vals)
        self.all_trained = True

    def predict(self, point):
        predictions = [tree.predict(point) for tree in self.trees]
        return np.mean(predictions)

    def bootstrap(self, points, labels, sample_size):
        num_points = len(points)
        if num_points == 0:
            return [], []

        bootstrap_points = []
        bootstrap_labels = []
        for choice in range(sample_size-1):
            point_idx = randint(0, num_points)
            bootstrap_points.append(points[point_idx])
            bootstrap_labels.append(labels[point_idx])
        return bootstrap_points, bootstrap_labels

    def possible_attr_vals(self, points, attributes):
        possible_vals = defaultdict(set)
        for attr in attributes:
            for point in points:
                if attr in point:
                    possible_vals[attr].add(point[attr])
        return possible_vals

class RandomTree:
    # add max_depth? and randomness value for bootstrapping? and num vals to pull from
    def __init__(self, false_child=None, true_child=None):
        self.attribute = None
        self.attr_val = None
        self.false_child = false_child
        self.true_child = true_child
        self.num_samples = 0
        self.prediction_val = None

    def train(self, points, labels, possible_vals):
        num_samples = len(points)
        if num_samples <= 5:
            # leaf node
            self.prediction_val = np.mean(labels)
            self.num_samples = num_samples
            return

        all_attributes = points[0].keys()
        subset_attrs = self.random_subset(all_attributes, 5)  # how many?
        split_data = self.split_by_best_feature(subset_attrs, points, labels, possible_vals)
        self.prediction_val = np.mean(labels)
        self.num_samples = num_samples
        self.attribute = split_data['best_attr']
        self.attr_val = split_data['best_val']
        true_points = split_data['true_points']
        false_points = split_data['false_points']
        true_labels = split_data['true_points']
        false_labels = split_data['false_points']
        self.true_child = RandomTree()
        self.true_child.train(true_points, true_labels)
        self.false_child = RandomTree()
        self.false_child.train(false_points, false_labels)

    def random_subset(self, attributes, sample_size):
        subset = []
        for choice in range(sample_size):
            attr_idx = randint(0, sample_size-1)
            subset.append(attributes[attr_idx])
        return subset

    def split_by_best_feature(self, points, attributes, labels, possible_vals):
        best_split_score = None
        best_attr = None
        best_val = None
        best_true_points = None
        best_false_points = None

        # get split score for each value for each attribute
        for attr in attributes:
            for val in possible_vals[attr]:
                true_points = []
                false_points = []
                true_labels = []
                false_labels = []
                for point_idx, point in enumerate(points):
                    # add point to true or false list depending on its attribute value
                    if attr in point:
                        if point[attr] <= val:
                            true_points.append(point)
                            true_labels.append(labels[point_idx])
                        else:
                            false_points.append(point)
                            false_labels.append(labels[point_idx])

                # get split score from true and false lists
                true_means = [np.mean(true_points)] * len(true_points)
                false_means = [np.mean(false_points)] * len(false_points)
                split_score = mean_squared_error(true_means, true_labels)*len(true_points)\
                    + mean_squared_error(false_means, false_labels)*len(false_points)

                # if split is better, save its score, left/right split, etc
                if best_split_score is None or split_score < best_split_score:
                    best_split_score = split_score
                    best_attr = attr
                    best_val = val
                    best_true_points = true_points
                    best_false_points = false_points

        split_data = {'true_points': best_true_points, 'false_points': best_false_points,
                      'best_attr': best_attr, 'best_val': best_val}
        return split_data

    def predict(self, point):
        if self.num_samples <= 5:
            return self.prediction_val
        else:
            if self.attribute in point:
                point_attr_val = point[self.attribute]
                if point_attr_val <= self.attr_val:
                    child = self.true_child
                else:
                    child = self.false_child

                return self.predict(child)

            else:
                return None

if __name__ == '__main__':

    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    # for future tests

    # regressor.fit(X_train, y_train)
    # y_pred = regressor.predict(X_test)
    #
    # print('Mean Abs Error', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Sq Error', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))