from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.utils.extmath import softmax
from DDAE.Data_block_construction import DBC
import pandas as pd
from DDAE.Adaptive_weight_adjust import AWA
from metric_learn import LMNN


class MainDDAE:
    def __init__(self, lb_column, maj_label=0, min_label=1, n_neighbors=3, max_iter=50,
                 cost_ratio=2, weight_loss_pull=0.2, unstable_ratio=0.2):
        """

        :param lb_column: target name
        :param maj_label: the label of majority
        :param min_label: the label of minority
        :param n_neighbors: number of the neighbors
        :param max_iter: maximal iteration
        :param cost_ratio: ratio between the cost of false negative and that of the false positive
        :param weight_loss_pull: relative weight between pull and push terms in large margin nearest neighbor algorithm
        :param unstable_ratio:
        """

        self.maj_label = maj_label
        self.min_label = min_label
        # number of features
        self.n_components = 0
        # description for last(label) column
        self.lb_column = lb_column
        # value of k for kNN
        self.n_neighbors = n_neighbors
        # maximal iteration
        self.max_iter = max_iter
        # default number of data block
        self.sigma_s = 1
        # default weight pair
        self.weight_pair = (1, 1)
        # default data block
        self.data_blocks = []
        #
        self.freq = {}
        # list for probability
        self.proba_k = []
        # store the base classifiers
        self.list_of_clfs = []
        # ratio between the cost of false negative and that of the false positive
        self.cost_ratio = cost_ratio
        #
        self.yscore = []
        # flag used to mark wheter the classifier has been fitted
        self.fitted = False
        # relative weight between pull and push terms in large margin nearest neighbor algorithm
        self.weight_loss_pull = weight_loss_pull
        #
        self.unstable_ratio = unstable_ratio

    def dbc_generation(self, X_train, y_train):
        """
        Generation of the set of data blocks
        :return: a set of data blocks
        """

        print('dbcing')
        self.data_blocks = [X_train]

        dbc = DBC(X_train=X_train, y_train=y_train, maj_label=self.maj_label, min_label=self.min_label,
                  lb_column=self.lb_column)
        self.sigma_s = dbc.get_sigma()
        self.data_blocks = dbc.dbc_construct()

        self.cost_ratio = dbc.get_N_maj() / dbc.get_N_min()

        print('dbc finish!')

    def awa_generation(self, X_train, y_train):  # X_te, y_te):
        """
        Generation of overall weight pair
        :param X_train: features of train set
        :param y_train: labels of train set
        :return: overall weight pair
        """
        print('awaing')
        awa = AWA(data_blocks=self.data_blocks,
                  columns=[self.lb_column],
                  label_tuple=(self.maj_label, self.min_label),
                  n_neighbors=self.n_neighbors,
                  X_train=X_train,
                  y_train=y_train,
                  sigma_s=self.sigma_s,
                  cost_ratio=self.cost_ratio,
                  unstable_ratio=self.unstable_ratio)

        self.weight_pair = awa.overall_weight_pair()

        print('awa finish!')

    # get the frequency of label for each sample
    def get_frequencyOfLabel(self, list_OF_base_classifier, X_test):
        """
        Generation of frequency for each label
        :param list_OF_base_classifier: list of base classifiers
        :param X_test: testing set
        :return:
        """

        for i in range(X_test.shape[0]):
            # Initialization: The i-th sample is classified as negative for zero times, and as positive for zero times
            self.freq[i] = [0, 0]

        cnt = 0
        for clf in list_OF_base_classifier:
            # predict test samples through kNN classifier trained by transformed data from each data block
            pred = clf.predict(X_test)
            # calculate the frequency of label for each sample
            for x in range(X_test.shape[0]):
                if pred[x] == self.maj_label:
                    self.freq[x][0] += 1
                else:
                    self.freq[x][1] += 1
            cnt += 1

    def predict_proba(self, X_test):
        """
        Calculation of the probability of predicting one sample as negative and positive respectively
        :return: a list of probabilities and each element is in form:
                [probability of predicting as negative, probability of predicting as positive]
        """

        for _ in self.freq:
            neg_ = (self.weight_pair[0] * self.freq[_][
                0])  # / (self.weight_pair[0] * self.freq[_][0] + self.weight_pair[1] * self.freq[_][1])
            pos_ = (self.weight_pair[1] * self.freq[_][
                1])  # / (self.weight_pair[0] * self.freq[_][0] + self.weight_pair[1] * self.freq[_][1])

            self.proba_k.append([neg_, pos_])

        self.proba_k = softmax(np.array(self.proba_k).astype(np.float), copy=False)

        return self.proba_k

    def el_generation(self):
        """
        Ultimate prediction based on major voting and the overall weight pair from awa component
        :return: a list of predictions
        """
        print('eling')
        f_s = []
        for m in self.freq:
            if self.freq[m][0] * self.weight_pair[0] < self.freq[m][1] * self.weight_pair[1]:
                f_s.append(self.min_label)
            else:
                f_s.append(self.maj_label)

        print('el finish!')
        return f_s

    def fit(self, X_train, y_train):

        """

        :param X_train:
        :param y_train:
        :return:
        """
        print('fitting!')
        self.n_components = X_train.shape[1]

        self.data_blocks = [pd.concat([X_train, y_train], axis=1).reset_index()]
        # print(self.data_blocks[0].head(10))
        print('the number of blocks before_dbc:', len(self.data_blocks))
        self.dbc_generation(X_train, y_train)
        print('the number of  blocks after dbc:', len(self.data_blocks))

        self.weight_pair = [1, 1]
        print('before_weight_pair', self.weight_pair)
        self.awa_generation(X_train, y_train)
        print('after_weight_pair', self.weight_pair)

        self.list_of_clfs = self.dsi_fit_transform(self.data_blocks)
        self.fitted = True
        return self

    def predict(self, X_test):
        """

        :param X_test:
        :return:
        """
        print('predicting')
        self.check_fitted()
        self.get_frequencyOfLabel(list_OF_base_classifier=self.list_of_clfs, X_test=X_test)

        return self.el_generation()

    def dsi_fit_transform(self, data_blocks):
        """
        Distance metrics learning of single data block
        :param data_blocks:
        :return: Predictions of test set based on single data block
        """
        cnt = 0
        clfs = []
        for _ in data_blocks:
            cnt += 1
            x = _.iloc[:, 1:-1]
            y = _.iloc[:, -1]
            lmnn = LMNN(k=self.n_neighbors, n_components=self.n_components, regularization=self.weight_loss_pull).fit(x,
                                                                                                                      y)
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=lmnn.get_metric())
            knn = knn.fit(x, y)

            clfs.append(knn)

        return clfs

    def check_fitted(self):
        if not self.fitted:
            raise NotFittedError("This estimator_ is not fitted yet. Call 'fit' with "
                                 "appropriate arguments before using this estimator_.")
