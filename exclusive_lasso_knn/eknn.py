import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io
from sklearn import linear_model
from sklearn import model_selection
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

class exclusive_lasso:
    def __init__(self, X, y,  groups_vect, lambda_):
        self.X = X
        self.y = y
        self.groups_vect = groups_vect
        self.lambda_ = lambda_ # cross_validate then pass into the function


    def regularized_least_square(self,alphas):
        """calculate the residual of X*/alphas compared to the test sample y

        Args:
            alphas (nx1 vector): coefficients vector

        Returns:
            integer: residual between X*/alphas and y
        """
        residual = self.y - (self.X @ alphas)
        return (np.sum(residual ** 2) + self.lambda_ * (np.sum(alphas ** 2))) # Square root of residuals plus the lasso penalty

    def lasso_optimize(self, x0):
        """argmin on the least squares to find optimal sparse alphas

        Args:
            x0 (float): nx1 vector initial values of alphas

        Returns:
            optimal alphas (as sklearn object) : nx1 optimal vector of coefficients
        """
        func = lambda x: self.regularized_least_square(x)
        return minimize(func, x0, method='SLSQP')


class EkNN_C:
    def __init__(self, X, coefs_vect, x_labels, k):
        self.coefs_vect = coefs_vect
        self.k = k
        self.x_labels = x_labels
        self.X = X
        
        # generate k largest coeffcients for the model
        coefs = self.coefs_vect
        coefs = coefs.tolist()
        coefs.sort(reverse=True)
        max_coefs = coefs[:self.k]
        k_largest_coefs_vect = self.coefs_vect

        for j in range(len(self.coefs_vect)):
            if k_largest_coefs_vect[j] not in max_coefs:
                k_largest_coefs_vect[j] = 0
                
        self.k_largest_coefs_vect  = k_largest_coefs_vect
    
    def class_coefs_vect(self, label_index):
        """represent coefficients according to a particular class labels 

        Args:
            label_index (label of a particular class): represent using integer values
            k_largest_coefs (float): nx1 sparse vector containing only k largest optimal coefficients
        """
        c_coefs = np.zeros(self.X.shape[1])
        for i in range(len(self.x_labels[label_index])):
            if self.x_labels[label_index,i] == 1:
                c_coefs[i] = self.k_largest_coefs_vect[i]
            else:
                c_coefs[i] = 0
        return c_coefs
    
        
    def predict(self):
        # choose k largest neighbor coeficients
        # k_largest_coefs_vect = self.k_largest_coefs_vect
        # k_largest_coefs vector represents only k largest coeficients
        
        # build coeficient vector for each class. Coeficients that
        ## are not k largest neighbors will be 0
        
        class_coefs = [] # list of m coeficient vector represent m classes
        for i in range(len(self.x_labels)):
            class_coefs.append(self.class_coefs_vect( i))
        
        coefs_l1_norm = [np.linalg.norm(class_coefs[i], ord=1) for i in range(len(self.x_labels))]
        for i in range(len(self.x_labels)):
            if np.sum(class_coefs[i]) == max(coefs_l1_norm):
                return i
        return 0
            
        
        #calculate L1-norm of coeficient vector
        
class EkNN_R:
    def __init__(self, X, y, coefs_vect, x_labels, k):
        self.coefs_vect = coefs_vect
        self.k = k
        self.x_labels = x_labels
        self.X = X
        self.y = y
        
        # construct nx1 vector containing k non-zero coeficients
        # which are k largest coefficients
        # Other coeficients that are not one of the k largest remains 0
        coefs = self.coefs_vect
        coefs = coefs.tolist()
        coefs.sort(reverse=True)
        max_coefs = coefs[:self.k]
        k_largest_coefs = self.coefs_vect

        for j in range(len(self.coefs_vect)):
            if k_largest_coefs[j] not in max_coefs:
                k_largest_coefs[j] = 0
                
        self.k_largest_coefs_vect  = k_largest_coefs
    
    def class_coefs_vect(self, label_index, k_largest_coefs):
        """represent coeficients according to a particular class labels 

        Args:
            label_index (_type_): _description_
            k_largest_coefs (_type_): vector containing only k largest optimal coeficients
        """
        c_coefs = np.zeros(self.X.shape[1])
        for i in range(len(self.x_labels[label_index])):
            if self.x_labels[label_index,i] == 1:
                c_coefs[i] = k_largest_coefs[i]
            else:
                c_coefs[i] = 0
        return c_coefs
    

    def create_obs_vects(self):
        """helper method for vector for each training data (each observation)
            to calculate the distance later.

        Returns:
            _list_: list containing n nx1 vectors representing n observations in the 
            training set
        """
        x_vects = []
        
        p = self.X.shape[0] # number of unknowns
        
        for obsv in range(self.X.shape[1]):
            res = np.zeros(p)
            for i in range(p):
                res[i] = self.X[i,obsv]
            x_vects.append(res)
        
        return x_vects

    def distance(self):
        """calculate distance from the test sample y to k nearest neighbors

        Args:
            self.x (_float_): nx1 vector representing 1 observation in the training set
            self.alphas (_float_): coefficients vector

        Returns:
            _float_: nx1 vector represents contains k non-zero distance value from 
            y to k nearest neighbors
        """
        x_vects = self.create_obs_vects()
        d_vect = np.zeros(self.X.shape[1])
        
        for i in range(self.X.shape[1]):
            if self.k_largest_coefs_vect[i] == 0:
                d_vect[i] = 0
            else: 
                d_vect[i] = np.linalg.norm(self.y - self.k_largest_coefs_vect[i] * x_vects[i])
                
        return d_vect
    
    def weights(self):
        """returns the weight corresponding to the distance of each observation to y

        Returns:
            _float_: nx1 vector shows weights of n observations to y
        """
        
        weights = np.zeros(self.X.shape[1])
        d_vect = self.distance()
        
        for i in range(self.X.shape[1]):
            if self.k_largest_coefs_vect[i]==0:
                weights[i] = -1
            elif np.amax(d_vect) != np.amin(d_vect):
                weights[i] = ((np.amax(d_vect) - d_vect[i])/(np.amax(d_vect)-np.amin(d_vect)))
            else:
                weights[i] = 1
        return weights
    
    def predict(self):
        """returns y's label according to 

        Returns:
            _integer_: integer encoding predicted class label of y (l_y)
        """
        W = self.weights()
        class_coefs_mat = np.zeros((self.x_labels.shape[0], self.x_labels.shape[1]))
        # class_coefs_list = []
        
        for i in range(self.x_labels.shape[0]):
            class_coefs_mat[i,:] = (self.class_coefs_vect(i, self.k_largest_coefs_vect))
        
        sum_coefs_weights = class_coefs_mat @ W
        # return W
        return np.where(sum_coefs_weights==max(sum_coefs_weights))[0][0]

class ExclusiveLassoKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_=1.0, group_num=1, k=3):
        self.lambda_ = lambda_
        self.group_num = group_num
        self.k = k
    
    def fit(self, X, x_labels):
        
        # X passed in have nxp dimension
        self.group_vect = self.group_encode(X) 
        self.X = np.transpose(X) # X_train
        self.x_labels = x_labels # y_train 
    
        class_names = [i+1 for i in range(len(np.unique(x_labels)))]
        
        self.x_labels = self.label_matrix(class_names, x_labels)

        self.is_fitted_ = True

        return self
    
    def predict(self, Y):
        # Y test sample
        preds = []
        
        # modify y's shape
        y_s = np.zeros((Y.shape[1], Y.shape[0]))
        for i in range(Y.shape[1]):
            for j in range(Y.shape[0]):
                y_s[i,j] = Y[j,i]
        
        xL2 = np.zeros(self.X.shape[1])
        for i in range(self.X.shape[1]):
            if np.random.rand() >= 0.5:
                xL2[i] = 1
            else:
                xL2[i] = 0
                
        for i in range(y_s.shape[0]):
            reg = exclusive_lasso(self.X, y_s[i], self.group_vect, self.lambda_)
            coefs = reg.lasso_optimize(xL2).x
            knn_R = EkNN_R(self.X, y_s[i], coefs, self.x_labels, self.k)
            preds.append(knn_R.predict())

        return np.array(preds)
    
    
    def label_matrix(self,label_names:list, labels):
        class_num = len(label_names)
        n_obs = labels.shape[0]
        label_mat = np.zeros((class_num, n_obs))
        for i in range(class_num):
            for ind,j in enumerate(labels):
                if j == label_names[i]:
                    label_mat[i,ind] = 1
                else:
                    label_mat[i,ind] = 0
        return label_mat
    
    
    def group_encode(self,X):
        # call before having X
        n = X.shape[1]
        group_pop = int(n/self.group_num)
        group_vect = []
        start = 0
        end = group_pop
        while end <= n:
            temp = np.zeros(X.shape[1])
            if end + group_pop > n:
                temp[start:n] = 1
                group_vect.append(temp)
                break
            else: temp[start:end] = 1
            start = end
            end += group_pop
            group_vect.append(temp)
        return np.array(group_vect)

    def score(self, Y, y_labels):
        """
        Calculate the accuracy of the classifier.
        """
        y_pred = self.predict(Y)
        return accuracy_score(y_labels, y_pred)  
            
        
        
        