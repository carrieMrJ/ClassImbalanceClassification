from metric_learn import LMNN
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class IterativeMetricLearning:
    def __init__(self, target_name, n_neighbors=5, regularization=0.5, base_classifier=KNeighborsClassifier(),
                 max_iter=1, label_tupel=(0, 1),
                 top_positive_number=3, top_negative_number=3, matching_ratio=0.8):

        """

        :param target_name:
        :param n_neighbors: number of the neighbors
        :param regularization: relative weight between pull and push terms in large margin nearest neighbor algorithm
        :param base_classifier:
        :param max_iter: maximal iteration for LMNN
        :param label_tupel: label of negative and positive in tuple
        :param top_positive_number: number of selected positive nearest neighbors
        :param top_negative_number: number of selected negative nearest neighbors
        :param matching_ratio:
        """
        self.y_train = None
        self.X_train = None
        self.regularization = regularization
        self.n_neighbors = n_neighbors
        self.base_classifier = base_classifier
        self.max_iter = max_iter
        self.n_positive = top_positive_number
        self.n_negative = top_negative_number
        self.target_name = target_name
        self.matching_ratio = matching_ratio
        self.predict_proba_list = []
        self.transformed_X_train = []
        self.label_tupel = label_tupel

    # iterative metric learning
    def data_space_metric_learning(self, x, y, testInstance):

        """
        data space metric learning by LMNN and selection of sub training set for the given testing sample
        :param x: features
        :param y: labels
        :param testInstance: testing sample
        :return: selected sub training set for the current sample
        """

        # distance metric learning by LMNN
        lmnn = LMNN(regularization=self.regularization, k=self.n_neighbors, max_iter=1)

        self.transformed_X_train = pd.DataFrame(lmnn.fit_transform(x, y), columns=x.columns)

        y.columns = [self.target_name]
        X_y_train = pd.concat([self.transformed_X_train, y], axis=1).reset_index(drop=True)

        pos_p = pd.DataFrame(X_y_train.loc[X_y_train[self.target_name] == self.label_tupel[1]])
        neg_p = pd.DataFrame(X_y_train.loc[X_y_train[self.target_name] == self.label_tupel[0]])

        # current testing sample
        testInstance = np.array(testInstance).reshape(1, -1)

        # find the positive neighbors
        clf_positive = KNeighborsClassifier(n_neighbors=self.n_positive)
        clf_positive.fit(pos_p.iloc[:, :-1], pos_p.iloc[:, -1])
        index_pos = clf_positive.kneighbors(testInstance, return_distance=False)
        pos_neighbors = pos_p.iloc[index_pos[0], :]

        # find the negative neighbors
        clf_negative = KNeighborsClassifier(n_neighbors=self.n_negative)
        clf_negative.fit(neg_p.iloc[:, :-1], neg_p.iloc[:, -1])
        index_neg = clf_negative.kneighbors(testInstance, return_distance=False)
        neg_neighbors = neg_p.iloc[index_neg[0], :]

        # set up the sub training set for the current testing sample
        sub_training_set = pd.concat([pos_neighbors, neg_neighbors], axis=0)

        return sub_training_set

    def data_matching(self, previous_set, cur_set, matching_ratio):
        """
        Compare the element of previous selected sub training set and that of the current selected sub training set
        :param previous_set: selected sub training set by the last iteration
        :param cur_set: selected sub training set by the current iteration
        :param matching_ratio: the given matching ratio
        :return:
        """
        cnt = 0
        previous_index = previous_set.index
        current_index = cur_set.index

        for i in previous_index:
            if i in current_index:
                cnt += 1

        # If more than a certain percentage of samples are selected again, it can be considered that the current neighborhood is stable
        if cnt / cur_set.shape[0] >= matching_ratio:

            return True
        else:
            return False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        print('k=', self.n_neighbors)
        predict_res = []
        predict_proba = []
        i = 0
        for index in range(X_test.shape[0]):

            test_sample = X_test.iloc[index, :]

            trigger = False
            cnt = 0
            previous_set = []
            curr_set = []
            while True:

                if trigger:
                    break

                if cnt == 0:
                    previous_set = self.data_space_metric_learning(self.X_train, self.y_train, test_sample)
                    cnt += 1
                    continue

                curr_set = self.data_space_metric_learning(self.transformed_X_train, self.y_train, test_sample)

                trigger = self.data_matching(previous_set, curr_set, self.matching_ratio)
                previous_set = curr_set
                cnt += 1

            final_set = curr_set
            lmnn = LMNN(regularization=self.regularization, k=self.n_neighbors, max_iter=1) \
                .fit(final_set.iloc[:, :-1], final_set.iloc[:, -1])

            self.base_classifier = self.base_classifier.set_params(**{'metric': lmnn.get_metric()})

            self.base_classifier.fit(final_set.iloc[:, :-1], final_set.iloc[:, -1])
            test_sample = np.array(test_sample).reshape(1, -1)
            predicted_label = self.base_classifier.predict(test_sample)[0]
            probability = self.base_classifier.predict_proba(test_sample)[0]
            predict_res.append(predicted_label)
            predict_proba.append(probability)
            i += 1

        self.predict_proba_list = predict_proba

        return np.array(predict_res)

    def predict_proba(self):
        return np.array(self.predict_proba_list)
