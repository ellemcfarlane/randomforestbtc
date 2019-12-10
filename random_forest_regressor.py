from random import randint
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from pickle_dataset import load_data_frame
import multiprocessing as mp
import line_profiler

# problems: gets attributes via list(points[0].keys), but each points may not have all attributes

# questions: max recursion depth sometimes reached for 200; not sure how many samples per leaf; can a leaf have
# no samples? hmm hm; should we shuffle before split; should we set seed? but -> non-random trees
# bc. how can we compare error rate as adjust hyper parameters
# WHY is points ever passed as None
# how do we deal with the random variation in MSE for predictions...?
# kind of slow esp. c.f. pre-built model

# should possible values be based purly on the data or range within the data?

# should we allow a child to have no samples if all splits yield 0 for one group? but then what happens
# if a test point IS to left or right? then what?

class RandomForestRegressor:
    """
    Class for building a random forest and using it to predict values from input data.
    """
    def __init__(self, num_estimators, pool_num_attributes):
        """
        Class for building a r
        :param num_estimators: number of trees in forest
        :param pool_num_attributes: number of random attributes for each node of a tree to consider
        """
        self.num_estimators = num_estimators
        self.pool_num_attributes = pool_num_attributes
        # create untrained forest
        self.trees = [RandomTree() for _ in range(num_estimators)]
        # set to True after all trees in forest are trained
        self.all_trained = False
    #@profile
    def build_forest(self, points, labels):
        """
        Builds a random forest trained on the given points with their respective classification labels.
        :param points: list of dictionaries where the key in each dictionary
            is an attribute category for the points.
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param sample_attr_size: int, number of random attributes to consider for each node
        Note: points must coincide with labels
        """
        len_points = len(points)
        len_labels = len(labels)
        if len_points != len_labels or len_points <= 0 or len_labels <= 0:
            raise Exception('Length of points must be non-zero and equal to length of labels.'
                            'Points length is {}, labels length is {}'.format(len_points, len_labels))
        # sample size for bootstrapping
        sample_size = len_points
        # all categories for each point
        attributes = list(points[0].keys())
        bs_points = []
        bs_labels = []
        bs_vals = []
        for _ in self.trees:
            bootstrap_points, bootstrap_labels = self.bootstrap(points, labels, sample_size)
            bs_points.append(bootstrap_points)
            bs_labels.append(bootstrap_labels)
            bs_vals.append(self.possible_attr_vals(bootstrap_points, attributes))
        # train each tree in forest with random bootstrap sample of size sample_size
        #count = 0
        # create multi-processing workers
        workers = mp.Pool(mp.cpu_count())
        task_results = []
        for tree, bootstrap_points, bootstrap_labels, possible_vals in zip(self.trees, bs_points, bs_labels, bs_vals):
            #bootstrap_points, bootstrap_labels = self.bootstrap(points, labels, sample_size)
            # get range of values for attributes
            #possible_vals = self.possible_attr_vals(bootstrap_points, attributes)
            #tree.train(bootstrap_points, bootstrap_labels, possible_vals, self.pool_num_attributes)
            task_results.append(workers.apply_async(RandomTree.train, (tree, bootstrap_points, bootstrap_labels, possible_vals, self.pool_num_attributes)))
            # count += 1
            # print(count)
        # get results
        new_trees = []
        count = 0
        for result in task_results:
            new_trees.append(result.get())
            print("tree", count)
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
        for choice in range(sample_size):
            # choose random index for a point (randint range is inclusive)
            point_idx = randint(0, num_points-1)
            # save point and its classification
            bootstrap_points.append(points[point_idx])
            bootstrap_labels.append(labels[point_idx])
        return bootstrap_points, bootstrap_labels

    def possible_attr_vals(self, points, attributes):
        """
        Returns dictionary where attribute category maps to set of possible values for that attribute
        :param points: list of dictionaries where the key in each dictionary
            is an attribute category for the points.
        :param attributes: list of possible categories a point could have
        :return: dictionary of sets
        """
        possible_vals = defaultdict(set)
        for attr in attributes:
            for point in points:
                if attr in point:
                    possible_vals[attr].add(point[attr])
        return possible_vals

class RandomTree:
    # add max_depth? and randomness value for bootstrapping? and num vals to pull from
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

    def train(self, points, labels, possible_vals, sample_attr_size):
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
        if num_samples <= 5:
            # predict based off of
            self.prediction_val = np.mean(labels) if labels else 0
            self.num_samples = num_samples
            return
        # get all possible attributes for the points
        all_attributes = list(points[0].keys())
        # get random sampling (no replacement) of attributes
        subset_attrs = self.random_subset(all_attributes, sample_attr_size)
        # divide samples into two groups via best split (best attribute-value combo)
        split_data = self.split_by_best_feature(subset_attrs, points, labels, possible_vals)
        # if no split_data, make current node a leaf
        if not split_data:
            self.prediction_val = np.mean(labels) if labels else 0
            self.num_samples = num_samples
            return
        # average classification values
        #self.prediction_val = np.mean(labels)
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
        self.true_child.train(true_points, true_labels, possible_vals, sample_attr_size)
        self.false_child = RandomTree()
        self.false_child.train(false_points, false_labels, possible_vals, sample_attr_size)
        return self

    def random_subset(self, attributes, sample_size):
        """
        Randomly samples from attributes and returns this subset
        :param attributes: list of possible categories a point could have
        :param sample_size: int specifying number of times to sample from attributes
        :return: list of possible categories
        """
        num_attributes = len(attributes)
        if sample_size > num_attributes:
            raise Exception("Sample size ({}) cannot exceed number of attributes ({}).".format(sample_size, num_attributes))
        subset = [attributes[randint(0, sample_size-1)] for _ in range(sample_size)]
        return subset

    def split_by_best_feature(self, attributes, points, labels, possible_vals):
        """
        Splits points via best attribute-value combo (where split score is lowest weighted MSE)
        :param points: list of dictionaries where the key in each dictionary
        :param attributes: list of possible categories a point could have
        :param labels: list of classifications (floats, booleans, etc) for each point
        :param possible_vals: dictionary where attribute category maps to set of possible values for that attribute
        :return: dictionary of two split groups and the attribute and value split was made on
        """
        best_split_score = None
        best_attr = None
        best_val = None
        best_true_points = None
        best_false_points = None
        best_true_labels = None
        best_false_labels = None

        num_points = len(points)
        if num_points <= 1:
            raise Exception("Cannot split just one value")
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


                # if split leaves a node with no samples, do not use this split
                if not true_points or not false_points:
                    continue
                # get split score from true and false lists
                true_means = [np.mean(true_labels)] * len(true_points)
                false_means = [np.mean(false_labels)] * len(false_points)

                true_score = mean_squared_error(true_means, true_labels)*len(true_points)
                false_score = mean_squared_error(false_means, false_labels)*len(false_points)
                split_score = (true_score + false_score)/num_points
                # if split is better (lower), save its score, left/right split, etc
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
    # points = [
    #     {'Date': '2016-07-17', 'Confirm Time': 7.76, 'Block Size': 0.68, 'Cost/TXN': 7.32, 'Difficulty': 213398925331.0,
    #      'TXN Vol': 129816925.1, 'Hash Rate': 1665474.54, 'Market Cap': 10453324633.1, 'Miners Rev': 1302095.02,
    #      'TXN/Block': 1160.45, 'Number of TXN': 182192.0, 'Unique Addresses': 356533.0, 'Total Bitcoin': 15763137.5,
    #      'TXN Fees': 50.67, 'Trade Vol': 13030366.55, 'TXN/Trade Ratio': 60.58214548260001, 'Price': 679.051,
    #      'Price Binary': 1},
    #     {'Date': '2016-07-16', 'Confirm Time': 8.98, 'Block Size': 0.79, 'Cost/TXN': 5.38, 'Difficulty': 213398925331.0,
    #      'TXN Vol': 187503475.3, 'Hash Rate': 1432095.94, 'Market Cap': 10514437454.2, 'Miners Rev': 1127082.34,
    #      'TXN/Block': 1600.59, 'Number of TXN': 216081.0, 'Unique Addresses': 367739.0, 'Total Bitcoin': 15761175.0,
    #      'TXN Fees': 55.22, 'Trade Vol': 26928546.83, 'TXN/Trade Ratio': 93.1540589114, 'Price': 663.541,
    #      'Price Binary': 0},
    #     {'Date': '2016-07-15', 'Confirm Time': 9.23, 'Block Size': 0.79, 'Cost/TXN': 5.49, 'Difficulty': 213398925331.0,
    #      'TXN Vol': 164674255.32, 'Hash Rate': 1506352.77, 'Market Cap': 10423167437.6, 'Miners Rev': 1175290.03,
    #      'TXN/Block': 1553.9, 'Number of TXN': 220655.0, 'Unique Addresses': 367953.0, 'Total Bitcoin': 15759487.5,
    #      'TXN Fees': 58.38, 'Trade Vol': 34318017.31, 'TXN/Trade Ratio': 76.3547780853, 'Price': 664.8760000000001,
    #      'Price Binary': 1},
    #     {'Date': '2016-07-14', 'Confirm Time': 10.73, 'Block Size': 0.81, 'Cost/TXN': 5.06,
    #      'Difficulty': 213398925331.0, 'TXN Vol': 187860897.96, 'Hash Rate': 1379055.35, 'Market Cap': 10366219391.2,
    #      'Miners Rev': 1070321.94, 'TXN/Block': 1677.63, 'Number of TXN': 218093.0, 'Unique Addresses': 367252.0,
    #      'Total Bitcoin': 15757725.0, 'TXN Fees': 53.55, 'Trade Vol': 53026862.77, 'TXN/Trade Ratio': 86.47388884360001,
    #      'Price': 659.6360000000001, 'Price Binary': 1},
    #     {'Date': '2016-07-13', 'Confirm Time': 9.26, 'Block Size': 0.77, 'Cost/TXN': 5.69, 'Difficulty': 213398925331.0,
    #      'TXN Vol': 193691457.42, 'Hash Rate': 1516960.89, 'Market Cap': 10682486713.9, 'Miners Rev': 1213263.1,
    #      'TXN/Block': 1531.7, 'Number of TXN': 219034.0, 'Unique Addresses': 379937.0, 'Total Bitcoin': 15756112.5,
    #      'TXN Fees': 51.17, 'Trade Vol': 48343367.13, 'TXN/Trade Ratio': 39.4733175548, 'Price': 653.934,
    #      'Price Binary': 0},
    #     {'Date': '2016-07-12', 'Confirm Time': 7.96, 'Block Size': 0.79, 'Cost/TXN': 5.49, 'Difficulty': 213398925331.0,
    #      'TXN Vol': 166782085.8, 'Hash Rate': 1516960.89, 'Market Cap': 10240153706.8, 'Miners Rev': 1162507.11,
    #      'TXN/Block': 1525.09, 'Number of TXN': 218089.0, 'Unique Addresses': 395742.0, 'Total Bitcoin': 15754325.0,
    #      'TXN Fees': 57.33, 'Trade Vol': 26402272.48, 'TXN/Trade Ratio': 57.0098951949, 'Price': 664.8389999999999,
    #      'Price Binary': 1}]
    #
    # labels = [num for num in range(len(points))]
    # regressor = RandomForestRegressor(10)
    # print(regressor.bootstrap(points, labels, 4))
    # regressor.build_forest(points, labels)

    dataset = load_data_frame('data_frames/old_btc_df.pkl')
    dataset['Price'] = dataset['Price'].shift(3)
    dataset.drop(0, axis=0, inplace=True)
    dataset.drop(1, axis=0, inplace=True)
    dataset.drop(2, axis=0, inplace=True)
    #dataset = dataset.sample(frac=1)
    y = dataset.iloc[:, 16].values
    dataset.drop(columns=['Price'], inplace=True)
    # print(dataset.head())
    # print(dataset.tail())
    #dataset = dataset[['Unique Addresses', 'Difficulty', 'Total Bitcoin']]
    X = dataset.to_dict('records')

    # split into training vs test attributes/labels
    train_sz = int(.8*len(X))

    X_train = X[:train_sz]
    y_train = y[:train_sz]
    X_test = X[train_sz:]
    y_test = y[train_sz:]

    # train
    # 20 trees, uses all 17 features for best split and n points for subsampling
    regressor = RandomForestRegressor(20, 17)
    regressor.build_forest(X_train, y_train)
    print(regressor.trees[0].true_child)
    y_pred = regressor.predict(X_test)

    #evaluate algo peformance
    print('Mean Abs Error', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Sq Error', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Sq Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))