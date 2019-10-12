from sklearn.datasets import load_breast_cancer
import numpy as np
from collections import Counter
import multiprocessing as mp
import scipy
import time

# Basic ID3 Tree
class id3_tree():
    'Implementation of ID3 Decision Tree in Python, majorly in NumPy'
    def __init__(self,least_children_num,verbose=True):
        self.least_children_num = least_children_num
        self.verbose = verbose
        
    def fit(self,tmp_x,tmp_y):
        def fit_tree(tmp_x,tmp_y):
        #     Exit Condition 0:
            # Exit Condition 1:
            if \
            len(tmp_y) < self.least_children_num or len(np.unique(tmp_y))==1:

                if self.verbose:
                    print('exit condition:')
                    print('tmp_y:')
                    print(tmp_y)

                mode_val = self.mode(tmp_y.flatten().tolist())
                return([np.nan, mode_val, np.nan, np.nan]) # Leaf Node: format [feat,splitval,]

            # Otherwise Split:
            if self.verbose:
                print("start....subset Y len {}".format(len(tmp_y)))


            split_row,split_col = self.decide_split_data(tmp_x,tmp_y)

            if not split_row and not split_col:
                mode_val = self.mode(tmp_y.flatten().tolist())
                return([np.nan, mode_val, np.nan, np.nan])

            if self.verbose:
                print("split on:")
                print(split_row,split_col)

            split_vec = tmp_x[:,split_col]
            split_val = tmp_x[split_row,split_col]
            # Recursively Split to left and right branches:
            left_ind = np.where(split_vec<split_val)[0].tolist()
            right_ind = np.where(split_vec>=split_val)[0].tolist()
            left_dat,left_y = tmp_x[left_ind,:],tmp_y[left_ind,]
            right_dat,right_y = tmp_x[right_ind,:],tmp_y[right_ind,]

            left_tree = fit_tree(left_dat,left_y)
            right_tree = fit_tree(right_dat,right_y)

            if isinstance(left_tree, list): # If list, tree len 1
                len_l_tree = 1
            else:
                len_l_tree = left_tree.shape[0] # If array, tree len >1

            root = [split_col,split_val,1,len_l_tree+1] # Format [split_col, split_val, left_tree_relative_idx, right_tree_relative_idx]
            return(np.vstack([root,left_tree,right_tree]))
        
        tree = fit_tree(tmp_x,tmp_y)
        self.tree = tree

    def decide_split_data(self,x,y):
        'Given subset of X,Y, search for the best splitting node based on: information gain'
        def entropy(tmp_y):
            'Key Metrics of building a decision tree. Specifically Shannon Entropy'
            tmp_ent = 0
            for uni_y in np.unique(tmp_y):
                p = len(tmp_y[tmp_y==uni_y])/len(tmp_y)
                tmp_ent -= (p*np.log2(p))
            return tmp_ent

        m,n = x.shape
        best_gain = 0
        split_row, split_col = None,None

        previous_entropy = entropy(y)
        for col in range(n):
            tmp_vec = x[:,col].ravel()

            for row in range(m):
                val = tmp_vec[row]
                # >= & < is my convention here:
                if val!=np.max(tmp_vec) and val!= np.min(tmp_vec):
                    left_b = np.where(tmp_vec<val)[0].tolist()
                    right_b = np.where(tmp_vec>=val)[0].tolist()

                    # new entropy is the weighted  average entropy from each of the subset
                    new_ent = \
                    (len(y[left_b])/len(y))*entropy(y[left_b]) + \
                    (len(y[right_b])/len(y))*entropy(y[right_b])

                    info_gain = previous_entropy - new_ent

                    if info_gain > best_gain:
                        split_row, split_col = row,col
                        best_gain = info_gain
                        if self.verbose:
                            print('better gain:{}'.format(best_gain))
                            print()

        return split_row, split_col
                
    def mode(self, x_list):
        'calculate the mode'
        return Counter(x_list).most_common(1)[0][0]

    def predict(self, tmp_test_array):
        'Wrap-up fun for prediction'
        def query(tree,tmp_test_array):
            'Test for single example'
            assert len(tmp_test_array.shape) == 2, "Make sure your test data is 2d array"

            if isinstance(tree,list):
                start_node = tree # only the 1 row in data
            else:
                start_node = tree[0,:] # Iteratively hit first row
            test_feat,test_val,left_tree_jump,right_tree_jump = start_node[0],start_node[1],start_node[2],start_node[3]
            # Exit Condition:
            if np.isnan(test_feat) and np.isnan(left_tree_jump) and np.isnan(right_tree_jump):
                pred = test_val
                return pred 
            #Test:
            if tmp_test_array[0,int(test_feat)] < test_val:
                # If <, go left branch:
                jump_loc = left_tree_jump
                pred = query(tree[int(jump_loc):,],tmp_test_array)
            else:
                # If >=, go right branch:
                jump_loc = right_tree_jump
                pred = query(tree[int(jump_loc):,],tmp_test_array)
            return pred
        assert len(tmp_test_array.shape) == 2, "Make sure your test data is 2d array"
        result = []
        for i in range(tmp_test_array.shape[0]):
            inp = tmp_test_array[i,:].reshape(1,-1)
            result.append(query(self.tree,inp))
        return result   



# RF using ID-3 tree:
class RandomForestClassification():
    """
    Python inplementation of random forest classifier 
    using id3 as the base tree
    with parallel processing
    """
    def __init__ (
        self,
        n_tree,
        min_leaf_num,  # to control overfit
        criteria = 'entropy', # currently only support entropy
        max_features = 'auto',# if max_feature = sqrt(number of features), otherwise will be proportion of features sampled
        n_workers = 1,
        verbose = True
        
    ):
        self.n_tree = n_tree
        self.min_leaf_num = min_leaf_num
        self.criteria = criteria
        self.max_features = max_features
        self.n_workers = n_workers
        self.verbose = verbose


    def fit_single(self,data):
        """
        Single ID3 Tree Fitting
        """
        X = data[0]
        y = data[1]
        tmp_X,tmp_y,feat_choose = self.random_find_feature(X,y)
        model = id3_tree(least_children_num = self.min_leaf_num,verbose=False)
        model.fit(tmp_X,tmp_y)
        return model,feat_choose

    def fit_rf(self,X,y):
        """
        Forest 
        """
        data = [X,y]
        with mp.Pool(self.n_workers) as p:
            model_list = p.map(self.fit_single,[data]*self.n_tree)
            
        self.model_list = model_list
        

    def predict_rf(self,X):
        """
        Forest Prediction
        taking the vote of each tree
        """
        result_list = []
        for model_stuff in self.model_list:
            print('.')
            single_model,single_feat_choose = model_stuff
            
            res = single_model.predict(X[:,single_feat_choose])
            result_list.append(res)
            
        return scipy.stats.mode(np.array(result_list),axis=0).mode.tolist()[0] # Take the vote
        
    
    def random_find_feature(self,X,y):
        """
        Randomly select subset of features for each tree
        """
    
        if self.max_features == 'auto':
            n_feat_dat = X.shape[1]
            n_feat_choose = int(round(np.sqrt(n_feat_dat)))
        else:
            n_feat_dat = X.shape[1]
            n_feat_choose = int(n_feat_dat*self.max_features)
            
        feat_choose = np.random.choice(range(n_feat_dat),size=n_feat_choose,replace=False).tolist()
        feat_choose = sorted(feat_choose) # Important to sort this in order otherwise will confuse the model
        print("feat_chosen:{}".format(feat_choose))


        return  X[:,feat_choose],y,feat_choose
