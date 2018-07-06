"""Numpy Implementation of ID3 Decision Tree Classifier."""
import numpy as np
from collections import Counter


class id3_Classifier():
    """
    The ID3 classifier is based on information gain to split.

    Usage:
    model = id3_tree_classifier(least_children_num = 4, verbose=True)
    model.fit(X_train,y)
    model.predict(X_test)
    """

    def __init__(self, least_children_num, verbose=True):
        """Constructor."""
        self.least_children_num = least_children_num
        self.verbose = verbose

    def fit(self, tmp_x, tmp_y):
        """Fit function."""
        def fit_tree(tmp_x, tmp_y):
            # Exit condition:
            if len(tmp_y) < self.least_children_num or len(np.unique(tmp_y)) == 1:

                if self.verbose:
                    print('exit condition:')
                    print('tmp_y:')
                    print(tmp_y)

                mode_val = self._mode(tmp_y.flatten().tolist())
                return([np.nan, mode_val, np.nan, np.nan])

            # Otherwise Split:
            if self.verbose:
                print("start....subset Y len {}".format(len(tmp_y)))
            split_row, split_col = self._decide_split(tmp_x, tmp_y)
            if not split_row and not split_col:
                print('no better split...return mode')
                mode_val = self._mode(tmp_y.flatten().tolist())
                return([np.nan, mode_val, np.nan, np.nan])

            if self.verbose:
                print("split on:")
                print(split_row, split_col)
            split_vec = tmp_x[:, split_col]
            split_val = tmp_x[split_row, split_col]
            left_ind = np.where(split_vec < split_val)[0].tolist()
            right_ind = np.where(split_vec >= split_val)[0].tolist()
            left_dat, left_y = tmp_x[left_ind, :], tmp_y[left_ind, ]
            right_dat, right_y = tmp_x[right_ind, :], tmp_y[right_ind, ]

            left_tree = fit_tree(left_dat, left_y)
            right_tree = fit_tree(right_dat, right_y)

            if isinstance(left_tree, list):
                len_l_tree = 1
            else:
                len_l_tree = left_tree.shape[0]

            root = [split_col, split_val, 1, len_l_tree + 1]
            return(np.vstack([root, left_tree, right_tree]))
        tree = fit_tree(tmp_x, tmp_y)
        self.tree = tree


    def _decide_split(self, x, y):
        """
        Given subset of X,Y,
        search for the best splitting node based on: information gain.
        """
        def _entropy(tmp_y):
            """Key Metrics of building a decision tree use Shannon Entropy."""
            tmp_ent = 0
            for uni_y in np.unique(tmp_y):
                p = len(tmp_y[tmp_y == uni_y]) / len(tmp_y)
                tmp_ent -= (p * np.log2(p))
            return tmp_ent

        m, n = x.shape
        best_gain = 0
        split_row, split_col = None, None

        previous_entropy = _entropy(y)
        for col in range(n):
            tmp_vec = x[:, col].ravel()
            for row in range(m):
                val = tmp_vec[row]
                # >= & < is the convention here:
                if val != np.max(tmp_vec) and val != np.min(tmp_vec):
                    left_b = np.where(tmp_vec < val)[0].tolist()
                    right_b = np.where(tmp_vec >= val)[0].tolist()

                    new_ent = (len(y[left_b]) / len(y)) * _entropy(y[left_b]) + \
                        (len(y[right_b]) / len(y)) * _entropy(y[right_b])
                    info_gain = previous_entropy - new_ent

                    if info_gain > best_gain:
                        split_row, split_col = row, col
                        best_gain = info_gain
                        if self.verbose:
                            print('better gain:{}'.format(best_gain))
                            print()
        return split_row, split_col

    def _mode(self, x_list):
        """Calculate the mode for splitting."""
        return Counter(x_list).most_common(1)[0][0]

    def predict(self, tmp_test_array):
        """Wrap-up fun for prediction."""
        def _query(tree, tmp_test_array):
            """Prediction for single example."""
            assert len(tmp_test_array.shape) == 2, \
                "Make sure your test data is 2d array"

            start_node = tree[0, :]
            test_feat, test_val, left_tree_jump, right_tree_jump = \
                start_node[0], start_node[1], start_node[2], start_node[3]

            if np.isnan(test_feat) and np.isnan(left_tree_jump) and \
                    np.isnan(right_tree_jump):

                pred = test_val
                return pred

            if tmp_test_array[0, int(test_feat)] < test_val:
                # If <, go left branch:
                jump_loc = left_tree_jump
                pred = _query(tree[int(jump_loc):, ], tmp_test_array)

            else:
                # If >=, go right branch:
                jump_loc = right_tree_jump
                pred = _query(tree[int(jump_loc):, ], tmp_test_array)

            return pred

        assert len(tmp_test_array.shape) == 2, \
            "Make sure test data is 2d-array"
        result = []

        for i in range(tmp_test_array.shape[0]):
            inp = tmp_test_array[i, :].reshape(1, -1)
            result.append(_query(self.tree, inp))
        return result
