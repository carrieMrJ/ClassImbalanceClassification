import math
import operator
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np


class AWA:

    def __init__(self, data_blocks, columns, label_tuple, n_neighbors, X_train, y_train, sigma_s, cost_ratio,
                 unstable_ratio):
        """

        :param data_blocks: Data Blocks generated through DBC Component
        :param columns: the name of class value column
        :param label_tuple: (maj_label, min_label) such as (0,1)
        :param n_neighbors: the number of neighbors
        :param X_train: the features of training set
        :param y_train: the class values of training set
        :param sigma_s: the number of data blocks
        :param cost_ratio: ratio between the cost of false negative and that of the false positive
        :param unstable_ratio:
        """
        self.data_blocks = data_blocks
        self.columns = columns
        self.X_y = pd.concat([X_train, y_train], axis=1).reset_index()
        self.label_tupel = label_tuple
        self.n_neighbors = n_neighbors
        self.X_train = X_train
        self.y_train = y_train
        self.sigma_s = sigma_s
        self.cost_ratio = cost_ratio
        self.unstable_ratio = unstable_ratio

    # Calculate distances between two instances based on Euclidean distances
    def distance_mertric(self, instance1, instance2, length):
        """
        Calculate distances between two instances based on Euclidean distances
        :param instance1: the first instance
        :param instance2: the second instance
        :param length: the number of features
        :return: Euclidean distances between two given instances
        """

        temp = 0
        for x in range(length):
            temp += pow((instance1[x] - instance2[x]), 2)
        distance = math.sqrt(temp)
        return distance

    # Find k neighbors for a test Instance in a single data block
    def getNeighbors(self, ith_data_block, testInstance):
        """
        Find k neighbors for a test Instance in a single data block
        :param ith_data_block:
        :param testInstance:
        :return: A list with k data samples as neighbors for the given test instance in the given data block
        """

        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        clf.fit(ith_data_block.iloc[:, 1:-1], ith_data_block.iloc[:, -1])
        testInstance = testInstance[1:-1]
        testInstance = np.array(testInstance).reshape(1, -1)

        neighbors_index = clf.kneighbors(testInstance, return_distance=False)

        return ith_data_block.iloc[neighbors_index[0], :]

    def get_class_votes(self, neighbors):

        """
        Determination of class distribution for the k neighbors
        :param neighbors: A list with k data samples
        :return:
        """

        # Count the number of neighbors for each class
        class_Votes = {self.label_tupel[0]: 0, self.label_tupel[1]: 0}

        for x in range(len(neighbors)):
            # print(neighbors.iloc[x,:])
            response = neighbors.iloc[x, 0]
            # print('response', response)

            if response in class_Votes:
                class_Votes[response] += 1

        sortedVotes = dict(sorted(class_Votes.items(), key=operator.itemgetter(1), reverse=True))

        return sortedVotes

    # Determination of unstable samples
    def generate_unstable(self, data, class_votes):

        """
        Determination of unstable samples
        :param data: all data samples that need to be verified
        :param class_votes: the number of majority and minority among the neighbors of each sample in data
        :return: a set of unstable samples
        """

        if self.n_neighbors % 2 == 0:
            threshold = 2
        else:
            threshold = 1

        unstable = pd.DataFrame(columns=data.columns[:])
        stable = pd.DataFrame(columns=data.columns[:])

        for i in range(data.shape[0]):

            pncd = abs(class_votes[i][self.label_tupel[0]] - class_votes[i][self.label_tupel[1]])

            if pncd > threshold:
                stable = stable.append(data.iloc[i, :])
            else:
                unstable = unstable.append(data.iloc[i, :])

        return unstable

    # Generation of unstable confusion matrix
    def get_unstable_Confusion_matrix(self, clf_fitted, unstable_samples):

        """
        :param clf_fitted: fitted kNN classifier
        :param unstable_samples: unstable samples need to be predicted
        :return: confusion matrix for unstable samples
        """

        unstable_predict = clf_fitted.predict(unstable_samples.iloc[:, 1:-1])

        un_c_matrix = confusion_matrix(unstable_samples.iloc[:, -1].values.astype('int').tolist(),
                                       unstable_predict.tolist())
        return un_c_matrix

    # Generation of weight threshold(Equation see section AWA(2))
    @property
    def get_Wt(self):
        """
        :return: weight threshold
        """
        temp = self.sigma_s / 2
        if self.sigma_s % 2 == 0:
            if (0.5 * self.sigma_s - 1) != 0:
                return (0.5 * self.sigma_s + 1) / (0.5 * self.sigma_s - 1)
            else:
                return 1
        elif temp >= 1:
            return (math.floor(self.sigma_s / 2) + 1) / math.floor(self.sigma_s / 2)
        else:
            return 1

    def adjust_weight(self, unstable_matrix, cost_ratio):
        """
        :param unstable_matrix: confusion matrix for unstable samples
        :param cost_ratio: the given cost num
        :return: weight pair for a single data block
        """
        # print(unstable_matrix)
        # default weight Wd=1
        default_weight = 1
        # default weight pair
        weight_pair = (1, 1)

        # calculate the value of three gains(Equation see section AWA(1))
        gain_mat = cost_ratio * (unstable_matrix[1][1] - unstable_matrix[1][0]) + (
                unstable_matrix[0][0] - unstable_matrix[0][1])
        gain_pos = cost_ratio * (unstable_matrix[1][1] + unstable_matrix[1][0]) + (
                -unstable_matrix[0][0] - unstable_matrix[0][1])
        gain_neg = cost_ratio * (-unstable_matrix[1][1] - unstable_matrix[1][0]) + (
                unstable_matrix[0][0] + unstable_matrix[0][1])

        # find the maximal value among these three gains
        gain_list = [gain_mat, gain_pos, gain_neg]
        gain_max = max(gain_list)
        max_idx = gain_list.index(gain_max)

        # update the weight pair
        if max_idx == 1:
            weight_pair = (default_weight, self.generate_non_default_weight())
        elif max_idx == 2:
            weight_pair = (self.generate_non_default_weight(), default_weight)

        return weight_pair

    # Generation of weight threshold(equation see section AWA(3))
    def generate_non_default_weight(self):
        """
        :return: weight threshold
        """
        return self.get_Wt + 0.01

    # Generation weight pair for the current data block
    def generate_weights(self, ith_data_block):

        """
        Generation weight pair for the current data block
        :param ith_data_block: current data block
        :return: weight pair for the current data block
        """

        neighbors = []
        # find neighbor lists for data sample of the current data block
        for i in (range(ith_data_block.shape[0])):
            # print('finding neighbors for', i, 'th sample')
            neighbors.append(pd.DataFrame(columns=self.columns))

            testInstance = ith_data_block.iloc[i, :]

            neighbors[i] = neighbors[i].append(self.getNeighbors(ith_data_block, testInstance))

        # get class votes for each data sample
        sortedVotes = []
        for i in range(len(neighbors)):
            sortedVotes.append(self.get_class_votes(neighbors[i]))

        # find unstable samples of the current data block
        unstable = self.generate_unstable(ith_data_block, sortedVotes)

        # initialize kNN classifier
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        # train the classifier using the current data block
        clf = clf.fit(ith_data_block.iloc[:, 1:-1], np.array(ith_data_block.iloc[:, -1], dtype=int))

        # generate unstable confusion matrix
        unstable_c_matrix = self.get_unstable_Confusion_matrix(clf, unstable)

        # get weight pair for current data block

        cur_weight_pair = self.adjust_weight(unstable_c_matrix, self.cost_ratio)

        return cur_weight_pair

    # Generation of overall weight pair
    def overall_weight_pair(self):
        """
        Generation of overall weight pair
        :return: ultimate overall weight pair
        """

        weight_pairs = {'default': 0, 'neg': 0, 'pos': 0}

        # generate weight pair for all data blocks
        for i in range(len(self.data_blocks)):

            weight = self.generate_weights(self.data_blocks[i])
            # count the number of each kind of weight pair
            if weight == (1, 1):
                weight_pairs['default'] += 1
            elif (weight[0] != 1) & (weight[1] == 1):
                weight_pairs['neg'] += 1
            else:
                weight_pairs['pos'] += 1
        print('weight_freq', weight_pairs)

        sum_ = (weight_pairs['default'] + weight_pairs['pos'] + weight_pairs['neg'])

        # the num between non-default weight pairs and all weight pairs
        ratio = (weight_pairs['pos'] + weight_pairs['neg']) / sum_
        print('num: (', weight_pairs['pos'], "+", weight_pairs['neg'], ') /', sum_, "=", ratio)

        # threshold is set to 0.2 in the experiment
        w_o = (1, 1)  # default
        w_n = self.generate_non_default_weight()

        if ratio >= self.unstable_ratio:
            if weight_pairs['neg'] > weight_pairs['pos']:
                w_o = (w_n, 1)
            else:
                w_o = (1, w_n)

        print('overall weight: ', w_o)
        print('overall finish')
        return w_o
