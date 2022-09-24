import math
import pandas as pd


# Class for Data Block Construction Component
class DBC:
    def __init__(self, X_train, y_train, maj_label, min_label, lb_column):
        """

        :param X_train:
        :param y_train:
        :param maj_label: the label of majority
        :param min_label: the label of minority
        :param lb_column: name of Class Value column
        """
        self.X_train = X_train
        self.y_train = y_train
        self.maj_label = maj_label
        self.min_label = min_label
        self.lb_column = lb_column
        self.N_maj = y_train.value_counts()[maj_label]
        self.N_min = y_train.value_counts()[min_label]

    def get_sigma(self):
        sigma_s = math.ceil(self.N_maj / self.N_min)

        return sigma_s

    def get_N_maj(self):
        return self.N_maj

    def get_N_min(self):
        return self.N_min

    def dbc_construct(self):

        print('#Maj:', self.N_maj)
        print('#Min:', self.N_min)
        sigma_s = self.get_sigma()

        print(f'sigma_s = {sigma_s}')


        maj_in_block = (self.N_maj // sigma_s) + 1

        print(f'number of samples in each buck = {maj_in_block}+{self.N_min}')

        dataset = pd.concat([self.X_train, self.y_train], axis=1).reset_index()


        idx_maj = []
        for i in range(sigma_s):
            idx_maj.append([])

        idx_min = []
        cnt = 0
        index_nr = 0

        # Filter index for majority and minority respectively
        for i, row in dataset.iterrows():

            if row[self.lb_column] == self.maj_label:
                if cnt < maj_in_block:
                    idx_maj[index_nr].append(i)
                else:
                    index_nr = index_nr + 1
                    cnt = 0
                    idx_maj[index_nr].append(i)
                cnt = cnt + 1
            else:
                idx_min.append(i)

        columns_ = dataset.columns[:]
        # set for majority
        S_maj = []
        S_min = pd.DataFrame(columns=columns_)
        for i in range(sigma_s):
            S_maj.append(pd.DataFrame(columns=columns_))
        for i in range(sigma_s):
            S_maj[i] = S_maj[i].append(dataset.iloc[idx_maj[i], :])

        # set for minority
        S_min = S_min.append(dataset.iloc[idx_min, :])

        # Data Blocks
        B = []
        for i in range(sigma_s):
            B.append(pd.DataFrame(columns=columns_))

        for i in range(sigma_s):
            B[i] = B[i].append(S_maj[i])
            B[i] = B[i].append(S_min)
            B[i] = pd.DataFrame(B[i])

        return B
