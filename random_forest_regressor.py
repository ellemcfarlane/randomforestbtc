import random
import numpy as np
from sklearn import metrics
from collections import defaultdict
import multiprocessing as mp
import line_profiler
import pickle
from matplotlib import pyplot as plt
import pandas as pd

class RandomForestRegressor:
    """
    Class for building a random forest and using it to predict values from input data.
    """
    def __init__(self, num_estimators):
        """
        Class for building a r
        :param num_estimators: number of trees in forest
        :param pool_num_attributes: number of random attributes for each node of a tree to consider
        """
        self.num_estimators = num_estimators
        # create untrained forest
        self.trees = [RandomTree() for _ in range(num_estimators)]
        # set to True after all trees in forest are trained
        self.all_trained = False

        self.random_state = 0
    #@profile
    def build_forest(self, points, labels, num_subset_points="all", num_subset_attributes="all"):
        """
        Builds a random forest trained on the given points with their respective classification labels.
        :param points: list of dictionaries where the key in each dictionary
            is an attribute category for the points.
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param sample_attr_size: int, number of random attributes to consider for each node
        Note: points must coincide with labels
        """
        # all categories for each point
        attributes = list(points[0].keys())
        num_attributes = len(attributes)

        num_points = len(points)
        num_labels = len(labels)
        if num_points != num_labels or num_points <= 0 or num_labels <= 0:
            raise Exception('Length of points must be non-zero and equal to length of labels.'
                            'Points length is {}, labels length is {}'.format(num_points, num_labels))
        # sample size for bootstrapping
        if num_subset_points == "all":
            num_subset_points = num_points
        else:
            if type(num_subset_points) is float:
                num_subset_points = int(num_subset_points * num_points)

        if num_subset_attributes == "all":
            num_subset_attributes = num_attributes
        else:
            if type(num_subset_attributes) is float:
                num_subset_attributes = int(num_subset_attributes * num_attributes)

        bs_points = []
        bs_labels = []
        for _ in self.trees:
            bootstrap_points, bootstrap_labels = self.bootstrap(points, labels, num_subset_points)
            bs_points.append(bootstrap_points)
            bs_labels.append(bootstrap_labels)
            self.random_state += 1
        # train each tree in forest with random bootstrap sample of size sample_size
        # count = 0
        # create multi-processing workers
        workers = mp.Pool(mp.cpu_count())
        task_results = []
        for tree, bootstrap_points, bootstrap_labels in zip(self.trees, bs_points, bs_labels):
            #bootstrap_points, bootstrap_labels = self.bootstrap(points, labels, sample_size)
            # get range of values for attributes
            #tree.train(bootstrap_points, bootstrap_labels, self.pool_num_attributes, attributes)
            task_results.append(workers.apply_async(RandomTree.train, (tree, bootstrap_points, bootstrap_labels, num_subset_attributes, attributes)))
            # count += 1
            # print(count)
        # get results
        new_trees = []
        count = 0
        for result in task_results:
            new_trees.append(result.get())
            print(count)
            count += 1
        self.trees = new_trees
        # close pool of workers
        workers.close()

        self.all_trained = True

    def predict(self, points):
        """
        Classifies given test points with random forest
        :param points: list of dictionaries of attributes and their corresponding values
        :return: list of floats (classifications for each point)
        """
        predictions = []
        # get prediction for each point
        for point in points:
            # average prediction from each tree in forest for final point prediction
            point_prediction = np.mean([tree.predict(point) for tree in self.trees])
            predictions.append(point_prediction)
        return predictions

    def bootstrap(self, points, labels, sample_size):
        """
        Randomly samples points with replacement.
        :param points: list of dictionaries where the key in each dictionary
            is an attribute category for the points.
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param sample_size: int for number of samples to select from points
        :return: tuple of list of points and their classification labels
        """
        num_points = len(points)
        # nothing to sample from
        if num_points == 0:
            return [], []
        bootstrap_points = []
        bootstrap_labels = []
        random.seed(self.random_state)
        for choice in range(sample_size):
            # choose random index for a point (randint range is inclusive)
            point_idx = random.randint(0, num_points-1)
            # save point and its classification
            bootstrap_points.append(points[point_idx])
            bootstrap_labels.append(labels[point_idx])
        self.random_state += 1
        return bootstrap_points, bootstrap_labels

class RandomTree:
    """
    Class for tree in Random Forest
    """
    def __init__(self, false_child=None, true_child=None):
        """
        :param false_child: child to go to if point is false for given attribute
        :param true_child: child to go to if point is false for given attribute
        """
        # node divides samples via attribute
        self.attribute = None
        # node divides sample via <= attr_val (create binary split)
        self.attr_val = None
        # false_child holds samples that are not <= attr_val
        self.false_child = false_child
        # true_child holds samples that are <= attr_val
        self.true_child = true_child
        # samples node holds that it will divide in two
        self.num_samples = 0
        # only set if node is leaf
        self.prediction_val = None

    #@profile
    def train(self, points, labels, sample_attr_size, all_attributes):
        """
        trains the tree node on the given points.
        :param points: list of dictionaries where the key in each dictionary
            is an attribute category for the points.
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param possible_vals: dictionary
        :param sample_attr_size: int, number of random attributes to consider for each node
        :return: dictionary where attribute category maps to set of possible values for that attribute
        """
        if not points or not labels:
            raise Warning("Must have at least 1 point and 1 label to train with.")
        num_samples = len(points)
        # create leaf node
        if num_samples <= 1:
            # predict based off of
            self.prediction_val = np.mean(labels) if labels else None
            self.num_samples = num_samples
            return
        # get random sampling (no replacement) of attributes
        subset_attrs = all_attributes  #self.random_subset(all_attributes, sample_attr_size)
        # divide samples into two groups via best split (best attribute-value combo)
        split_data = self.split_by_best_feature(points, labels)
        # if no split_data, make current node a leaf
        if not split_data:
            self.prediction_val = np.mean(labels) if labels else 0
            self.num_samples = num_samples
            return
        # node now knows how many samples it holds, etc
        self.num_samples = num_samples
        self.attribute = split_data['best_attr']
        self.attr_val = split_data['best_val']
        # get both groups from the split
        true_points = split_data['true_points']
        false_points = split_data['false_points']
        true_labels = split_data['true_labels']
        false_labels = split_data['false_labels']
        # create true and false children for node each holding their respective samples based off of split
        self.true_child = RandomTree()
        self.true_child.train(true_points, true_labels, sample_attr_size, all_attributes)
        self.false_child = RandomTree()
        self.false_child.train(false_points, false_labels, sample_attr_size, all_attributes)
        return self

    def random_subset(self, attributes, sample_size):
        """
        Randomly samples (without replacement) from attributes and returns this subset
        :param attributes: list of possible categories a point could have
        :param sample_size: int specifying number of times to sample from attributes
        :return: list of possible categories
        """
        #seed(99)
        num_attributes = len(attributes)
        if sample_size > num_attributes:
            raise Exception("Sample size ({}) cannot exceed number of attributes ({}).".format(sample_size, num_attributes))
        subset = [attributes[random.randint(0, sample_size-1)] for _ in range(sample_size)]
        return subset

    def split_by_best_feature(self, points, labels):
        """
        Splits points via best attribute-value combo (where split score is lowest weighted MSE)
        :param points: list of dictionaries where the key in each dictionary
        :param attributes: list of possible categories a point could have
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param possible_vals: dictionary where attribute category maps to set of possible values for that attribute
        :return: dictionary of two split groups and the attribute and value split was made on
        """
        combos_seen = set()
        best_split_score = None
        best_attr = None
        best_val = None
        best_true_points = None
        best_false_points = None
        best_true_labels = None
        best_false_labels = None
        scores = []
        num_points = len(points)

        if num_points <= 1:
            raise Exception("Cannot split just one value")
        for point in points:
            for attr, val in point.items():
                if (attr, val) in combos_seen:
                    continue
                combos_seen.add((attr, val))
                true_points = []
                false_points = []
                true_labels = []
                false_labels = []
                for point_idx, other_point in enumerate(points):
                    # add point to true or false list depending on its attribute value
                    if attr in other_point:
                        if other_point[attr] <= val:
                            true_points.append(other_point)
                            true_labels.append(labels[point_idx])
                        else:
                            false_points.append(other_point)
                            false_labels.append(labels[point_idx])

                # if split leaves a node with no samples, do not use this split
                if not true_points or not false_points:
                    continue
                # get split score from true and false lists
                true_means = [np.mean(true_labels)] * len(true_points)
                false_means = [np.mean(false_labels)] * len(false_points)

                true_score = metrics.mean_squared_error(true_means, true_labels) * len(true_points)
                false_score = metrics.mean_squared_error(false_means, false_labels) * len(false_points)
                split_score = (true_score + false_score) / num_points
                # if split is better (lower), save its score, left/right split, etc
                scores.append((attr, val, split_score))
                if best_split_score is None or split_score < best_split_score:
                    best_split_score = split_score
                    best_attr = attr
                    best_val = val
                    best_true_points = true_points
                    best_false_points = false_points
                    best_true_labels = true_labels
                    best_false_labels = false_labels
        split_data = {'true_points': best_true_points, 'true_labels': best_true_labels,
                      'false_labels': best_false_labels, 'false_points': best_false_points,
                      'best_attr': best_attr, 'best_val': best_val}
        return split_data if best_split_score else None

    def predict(self, point):
        """
        Classifies given test point based on its attributes
        :param point: dictionary of attributes and their corresponding values
        :return: float
        """
        if self.prediction_val is not None:
            return self.prediction_val
        else:
            if self.attribute in point:
                point_attr_val = point[self.attribute]
                # go to child based off point's attribute value and node's split-value
                child = self.true_child if point_attr_val <= self.attr_val else self.false_child
                return child.predict(point)
            # point does not have node's split-attribute, so cannot be evaluated
            else:
                return None


if __name__ == '__main__':

    # Petroleum
    # dataset = pd.read_csv('raw_csvs/petrol_consumption.csv', low_memory=False).sample(frac=1, random_state=0)
    # y = dataset.iloc[:, 4].values
    # dataset.drop(dataset.columns[4], axis=1, inplace=True)

    #############################################################################

    # Old BTC

    dataset = pd.read_csv('raw_csvs/df_final.csv', low_memory=True)
    dataset['Price'] = dataset['Price'].shift(3)
    dataset.drop(0, axis=0, inplace=True)
    dataset.drop(1, axis=0, inplace=True)
    dataset.drop(2, axis=0, inplace=True)
    y_orig = dataset.loc[:, 'Price'].values[::-1]
    X_orig = dataset.drop(columns=['Price', 'Date', 'Unnamed: 0'])\
        .reindex(index=dataset.index[::-1]).to_dict('records')
    dataset = dataset.sample(frac=1, random_state=0)

    y = dataset.loc[:, 'Price'].values
    dataset.drop(columns=['Price', 'Date', 'Unnamed: 0'], inplace=True)
    print(dataset.head())

    #############################################################################

    # New BTC

    # dataset = pd.read_csv('raw_csvs/bitcoin_final.csv', low_memory=True)
    # dataset['market_price'] = dataset['market_price'].shift(-1)
    # dataset.drop(len(dataset) - 1, axis=0, inplace=True)
    # y_orig = dataset.loc[:, 'market_price'].values[::-1]
    # X_orig = dataset.drop(columns=['market_price', 'date', 'Unnamed: 0'])\
    #     .reindex(index=dataset.index[::-1]).to_dict('records')
    # print(dataset.head())
    # dataset = dataset.sample(frac=1, random_state=0)
    # y = dataset.loc[:, 'market_price'].values
    # dataset.drop(columns=['market_price', 'date', 'Unnamed: 0'], inplace=True)
    # print(dataset.head())

    #############################################################################

    # Train Forest

    X = dataset.to_dict('records')

    # split into training vs test attributes/labels
    train_sz = int(.8*len(X))

    X_train = X[:train_sz]
    y_train = y[:train_sz]
    X_test = X[train_sz:]
    y_test = y[train_sz:]
    print("train", y_train[:10])

    # train
    #20 trees, uses all features for best split and n points for subsampling
    # regressor = RandomForestRegressor(1)
    # regressor.build_forest(X_train, y_train)

    # with open('final_forest_2.pkl', 'wb') as file:
    #     pickle.dump(regressor, file)
    #
    with open('final_forest.pkl', 'rb') as file:
        regressor = pickle.load(file)

    y_pred = regressor.predict(X_test)
    y_train_pred = regressor.predict(X_train)
    y_full = regressor.predict(X_orig)

    print("y-pred", y_pred[-10:])
    print("x", X_orig[:1])

    #############################################################################

    # Evaluate algo performance

    errors = abs(y_pred - y_test)
    map = 100 * np.mean(errors / y_test)
    accuracy = 100 - map

    print('Performance')
    print('Average Error: ${:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print()
    print('Mean Abs Error', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Sq Error', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    #############################################################################

    # Visualize results
    x_test = range(len(y_pred))
    x_train = range(len(y_train_pred))
    x_full = range(len(y_pred) + len(y_train_pred))

    fig, axs = plt.subplots(3)
    axs[0].set_title('Bitcoin Price RF Predictions')
    axs[0].scatter(x_train, y_train, c='black')
    axs[0].scatter(x_train, y_train_pred, c='red', marker=".")

    axs[1].scatter(x_test, y_test, c='black')
    axs[1].scatter(x_test, y_pred, c='red', marker=".")

    axs[2].scatter(x_full, y_orig, c='black')
    axs[2].plot(x_full, y_full, c='red')
    axs[2].set_xlabel('Relative dates from 2012-2016')

    plt.show()
    print("random end state", regressor.random_state)


