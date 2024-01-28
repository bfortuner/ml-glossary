import numpy as np
from scipy import stats
from abc import ABCMeta
from typing import List


class TreeNode:
    def __init__(self, data_idx, depth, child_lst=None):
        """
        Initialize a TreeNode object.

        Parameters:
        - data_idx (int): Index of the data associated with the node.
        - depth (int): Depth of the node in the tree.
        - child_lst (list, optional): List of child nodes. Defaults to an empty list.
        """
        self.data_idx = data_idx
        self.depth = depth
        self.child = child_lst if child_lst is not None else []
        self.label = None
        self.split_col = None
        self.child_cate_order = None

    def set_attribute(self, split_col, child_cate_order=None):
        """
        Set the attributes of the node.

        Parameters:
        - split_col: Column used for splitting the data.
        - child_cate_order (list, optional): Order of child categories. Defaults to None.
        """
        self.split_col = split_col
        self.child_cate_order = child_cate_order

    def set_label(self, label):
        """
        Set the label for the node.

        Parameters:
        - label: The label to be assigned to the node.
        """
        self.label = label



class DecisionTree(metaclass=ABCMeta):
    def __init__(self, max_depth, min_sample_leaf, min_split_criterion=1e-4, verbose=False):
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf
        self.verbose = verbose
        self.min_split_criterion = min_split_criterion
        self.root = None
        self.data = None
        self.labels = None
        self.feature_num = None

    def fit(self, X, y):
        """
        X: train data, dimensition [num_sample, num_feature]
        y: label, dimension [num_sample, ]
        """
        self.data = X
        self.labels = y
        num_sample, num_feature = X.shape
        self.feature_num = num_feature
        data_idx = list(range(num_sample))
        self.root = TreeNode(data_idx=data_idx, depth=0, child_lst=[])
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.depth>self.max_depth or len(node.data_idx)==1:
                self.set_label(node)
            else:
                child_nodes = self.split_node(node)
                if not child_nodes:
                    self.set_label(node)
                else:
                    queue.extend(child_nodes)

    def predict(self, X):
        num_sample, num_feature = X.shape
        labels = []
        for idx in range(num_sample):
            x = X[idx]
            node = self.root
            while node.child:
                node = self.get_nex_node(node, x)
            labels.append(node.label)
        return labels

    @classmethod
    def get_split_criterion(self, node, child_node_lst):
        pass

    def set_label(self, node):
        target_Y = self.labels[node.data_idx]
        target_label = stats.mode(target_Y).mode[0]
        node.set_label(label=target_label)

    @classmethod
    def split_node(self, node):
        pass

    @classmethod
    def get_nex_node(self, node, x):
        pass


class ID3DecisionTree(DecisionTree):

    def split_node(self, node):
        child_node_lst = []
        child_cate_order = []
        informatin_gain = 0
        split_col = None
        for col_idx in range(self.feature_num):
            current_child_cate_order = list(np.unique(self.data[node.data_idx][:, col_idx]))
            current_child_node_lst = []
            for col_value in current_child_cate_order:
                data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] == col_value))
                current_child_node_lst.append(
                    TreeNode(
                        data_idx=data_idx,
                        depth=node.depth+1
                    )
                )
            current_gain = self.get_split_criterion(node, current_child_node_lst)
            if current_gain > informatin_gain:
                informatin_gain = current_gain
                child_node_lst = current_child_node_lst
                child_cate_order = current_child_cate_order
                split_col = col_idx
        if informatin_gain<self.min_split_criterion:
            return
        else:
            node.child = child_node_lst
            node.set_attribute(split_col=split_col, child_cate_order=child_cate_order)
            return child_node_lst

    def get_split_criterion(self, node, child_node_lst):
        total = len(node.data_idx)
        split_criterion = 0
        for child_node in child_node_lst:
            impurity = self.get_impurity(child_node.data_idx)
            split_criterion += len(child_node.data_idx) / float(total) * impurity
        return split_criterion

    def get_impurity(self, data_ids):
        target_Y = self.labels[data_ids]
        total = len(target_Y)
        unique, count = np.unique(target_Y, return_counts=True)
        res = 0
        for c in count:
            p = float(c)/total
            res -= p*np.log(p)
        return res

    def get_nex_node(self, node, x):
        try:
            next_node = node.child[node.child_cate_order.index(x[node.split_col])]
        except:
            next_node = node.child[0]
        return next_node


class C45DecisionTree(ID3DecisionTree):

    def get_split_criterion(self, node, child_node_lst):
        total = len(node.data_idx)
        split_criterion = 0
        for child_node in child_node_lst:
            impurity = self.get_impurity(child_node.data_idx)
            split_criterion += len(child_node.data_idx) / float(total) * impurity
        intrinsic_value = self._get_intrinsic_value(node, child_node_lst)
        split_criterion= split_criterion/intrinsic_value
        return split_criterion

    def _get_intrinsic_value(self, node, child_node_lst):
        total = len(node.data_idx)
        res = 0
        for n in child_node_lst:
            frac = len(n.data_idx) / float(total)
            res -=  frac * np.log(frac)
        return res


class CART(DecisionTree):

    def __init__(self, max_depth, min_sample_leaf, split_criterion="gini", tree_type="classification", min_split_criterion=1e-4, verbose=False):
        super(CART, self).__init__(max_depth=max_depth, min_sample_leaf=min_sample_leaf, min_split_criterion=min_split_criterion
                                   , verbose=verbose)
        self.tree_type = tree_type
        self.split_criterion = split_criterion
        assert self.split_criterion in ["gini", "entropy"]
        assert self.tree_type in ["classification", "regression"]

    def split_node(self, node: TreeNode) -> List[TreeNode]:
        child_node_lst = []
        child_cate_order = None
        gini_index = float("inf")
        split_col = None
        for col_idx in range(self.feature_num):
            current_child_cate_order = list(np.unique(self.data[node.data_idx][:, col_idx]))
            current_child_cate_order.sort()
            for col_value in current_child_cate_order:
                left_data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] <= col_value))
                right_data_idx = np.intersect1d(node.data_idx, np.where(self.data[:, col_idx] > col_value))
                current_child_node_lst = []
                if len(left_data_idx) != 0:
                    left_tree = TreeNode(
                            data_idx=left_data_idx,
                            depth=node.depth+1,
                        )
                    current_child_node_lst.append(left_tree)
                if len(right_data_idx) != 0:
                    right_tree = TreeNode(
                            data_idx=right_data_idx,
                            depth=node.depth+1,
                        )
                    current_child_node_lst.append(right_tree)
                current_gini_index = self.get_split_criterion(node, current_child_node_lst)
                if current_gini_index < gini_index:
                    gini_index = current_gini_index
                    child_node_lst = current_child_node_lst
                    child_cate_order = col_value
                    split_col = col_idx
        node.child = child_node_lst
        node.set_attribute(split_col=split_col, child_cate_order=child_cate_order)
        return child_node_lst

    def get_split_criterion(self, node, child_node_lst):
        total = len(node.data_idx)
        split_criterion = 0
        for child_node in child_node_lst:
            impurity = self.get_impurity(child_node.data_idx)
            split_criterion += len(child_node.data_idx) / float(total) * impurity
        return split_criterion

    def get_impurity(self, data_ids):
        target_y = self.labels[data_ids]
        total = len(target_y)
        if self.tree_type == "regression":
            res = 0
            mean_y = np.mean(target_y)
            for y in target_y:
                res += (y - mean_y) ** 2 / total
        elif self.tree_type == "classification":
            if self.split_criterion == "gini":
                res = 1
                unique_y = np.unique(target_y)
                for y in unique_y:
                    num = len(np.where(target_y==y)[0])
                    res -= (num/float(total))**2
            elif self.split_criterion == "entropy":
                unique, count = np.unique(target_y, return_counts=True)
                res = 0
                for c in count:
                    p = float(c) / total
                    res -= p * np.log(p)
        return res

    def get_nex_node(self, node: TreeNode, x: np.array):
        col_value = x[node.split_col]
        if col_value> node.child_cate_order:
            index = 1
        else:
            index = 0
        return node.child[index]


if __name__ == "__main__":
    # ID3: only categorical features
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, mean_squared_error
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    dataset = datasets.load_iris()


    # #############################
    # ========== Config ==========
    # #############################
    all_categorical_feature = True
    max_depth = 3
    min_sample_leaf = 4
    split_criterion = "entropy"
    # tree_type = "classification"
    tree_type = "regression"
    # ###########################

    # convert continuous feature to categorical features
    if all_categorical_feature:
        f = lambda x: int(x)
        func = np.vectorize(f)
        X = func(dataset.data)
    else:
        X = dataset.data

    Y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    if tree_type == "classification":
        model = DecisionTreeClassifier(criterion=split_criterion, max_depth=max_depth, min_samples_leaf=min_sample_leaf)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_sample_leaf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if tree_type == "classification":
        print(classification_report(y_true=y_test, y_pred=y_pred))
    else:
        print(mean_squared_error(y_test, y_pred))
    #
    # model = ID3DecisionTree(max_depth=max_depth, min_sample_leaf=min_sample_leaf, verbose=True)
    # model = C45DecisionTree(max_depth=max_depth, min_sample_leaf=min_sample_leaf, verbose=True)
    model = CART(max_depth=max_depth, min_sample_leaf=min_sample_leaf, verbose=True, tree_type=tree_type, split_criterion=split_criterion)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if tree_type == "classification":
        print(classification_report(y_true=y_test, y_pred=y_pred))
    else:
        print(mean_squared_error(y_test, y_pred))
