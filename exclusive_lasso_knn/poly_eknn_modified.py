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
from sklearn.cluster import KMeans

# same as eknn_poly
class poly_regression: 
    def __init__(self, X, y, groups_vect, lambda_):
        self.X = X
        self.y = y
        self.groups_vect = groups_vect
        # self.lambda_1 = lambda_1
        self.lambda_ = lambda_
        
    def regularized_least_square(self, coefficients):
        
        # coefficients vector is alphas stacked upon betas
        alphas = coefficients[0:self.X.shape[1]]
        betas = coefficients[self.X.shape[1]:]
        grouped_alphas = self.groups_vect @ np.absolute(alphas)
        grouped_betas = self.groups_vect @ np.absolute(betas)
        residual= self.y + self.y**2 - (self.X @ alphas + self.X**2 @ betas)
        return (np.sum(residual**2) + self.lambda_ * (np.sum(grouped_alphas ** 2)) + self.lambda_ * (np.sum(grouped_betas ** 2)) )
    
    def lasso_optimize(self, x0):
        func = lambda stacked_coefficients: self.regularized_least_square(stacked_coefficients)
        return minimize(func, x0, method='SLSQP')
    
class single_polynomial_EkNN_R:
    def __init__(self, X, y, alphas, betas, x_labels_encode, k):

        self.k = k
        self.X = X #pxn matrix
        self.y = y # px1 vector
        self.x_labels_encode = x_labels_encode 
    
        sum_coefs = alphas + betas
        sum_coefs_list = sum_coefs.tolist()
        sum_coefs_list.sort(reverse=True)
        # print(sum_coefs_list)
        max_sum_coefs_list = sum_coefs_list[0:self.k]
        # print(sum_coefs_list)
        
        k_largest_sum_coefs = alphas + betas
        k_indices = []
        
        for j in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[j] not in max_sum_coefs_list: # what if 2 or more coefficients have the same value?
                k_largest_sum_coefs[j] = 0
            else: k_indices.append(j)
        self.k_indices = k_indices
        
        # diagonal matrix with alphas and betas on the diagonal
        self.alphas_diag_matrix = np.diag(alphas) 
        self.betas_diag_matrix = np.diag(betas)

        
        # nx1 vector to calculate the label ly = w . (alpha+beta) 
        self.k_largest_sum_vect = k_largest_sum_coefs
        
    def class_coefs_matrix(self): # can we replace the loops ?

        class_coefs_matrix = np.zeros((self.x_labels_encode.shape[0],self.X.shape[1])) # m classes x n number of observations 
        
        for i in range(len(self.x_labels_encode)):
            for j in range(self.X.shape[1]):
                if self.x_labels_encode[i, j] == 1:
                    class_coefs_matrix[i, j] = self.k_largest_sum_vect[j]
                else:
                    class_coefs_matrix[i, i] = 0
        return class_coefs_matrix
        
    def distance(self):
        # calculate betas.x matrix
        beta_x2_matrix = (self.X ** 2) @ self.betas_diag_matrix #pxn matrix
        alpha_x_matrix = self.X @ self.alphas_diag_matrix # pxn matrix
        
        #create y+y^2 nx1 matrix 
        ys = np.zeros((self.X.shape[1], self.X.shape[0])) # nxp matrix
        for i in range(self.X.shape[1]):
            ys[i,:] = self.y + self.y**2
        ys = np.transpose(ys) # pxn
        terms = beta_x2_matrix + alpha_x_matrix
        comp_matrix = ys - terms
        
        squared_comp = np.transpose(comp_matrix) @ comp_matrix
        d = np.sqrt(np.diag(squared_comp))
        return d
        
    def weight(self):
        distance = self.distance()
        if np.amax(distance) == np.amin(distance):
            return np.diag(np.ones(self.X.shape[1]))

        else:
            w = (np.amax(distance) / (np.amax(distance) - np.amin(distance))) * np.ones(self.X.shape[1]) - 1/(np.amax(distance) - np.amin(distance)) * distance
            return w
    
    def classify(self):
        class_coefs_matrix = self.class_coefs_matrix() # mxn
        w = self.weight() 
        weights_coefs_product = class_coefs_matrix @ w
        print(weights_coefs_product, weights_coefs_product.shape)

        return np.where(weights_coefs_product == max(weights_coefs_product))[0][0]
        # consider when the coefficients are negative,which one will we pick 0 or tat negative number
        # if so, when constructing the class_coef_vect we can set the 0 to -inf
        
class polynomial_EkNN_R_classifier:
    def __init__(self, lambda_=1.0,  group_num=1, k=3):
        # self.lambda_1 = lambda_1
        self.lambda_ = lambda_
        self.group_num = group_num
        self.k = k
    
    def fit(self, X_train, y_train):
        self.group_vect = self.group_encode(X_train)
        self.X_train = np.transpose(X_train)
        class_names = [i for i in range(len(np.unique(y_train)))]
        
        self.y_train = self.label_matrix(class_names, y_train)
        self.is_fitted_ = True
        return self
        
    def group_encode(self, X): #
        # sth
        kmeans= KMeans(n_clusters = self.group_num, random_state=0, n_init="auto").fit(X)
        result = kmeans.labels_
        group_vect = np.zeros((self.group_num, len(result)))
        for i in range(self.group_num):
            for j in range(len(result)):
                if result[j] == i:
                    # print(result[j],"and ",i )
                    group_vect[i,j] = 1
        # print (group_vect.shape)
        return group_vect

        
        
    def label_matrix(self, label_names:list, labels): # can we change this for loops? 
        """_summary_

        Args:
            label_names (list): _description_
            labels (_numpy array_): nx1 matrix containing labels of the training observations

        Returns:
            _type_: _description_
        """
        m = len(label_names) # number of class labels
 
        n = labels.shape[0] # number of training observations
        label_mat = np.zeros((m, n))
        
        for i in range(m):
            for ind, j in enumerate(labels):
                if j == label_names[i]:
                    label_mat[i,ind] = 1
                else:
                    label_mat[i,ind] = 0
                    
        return label_mat

    def predict(self, X_test):
        # create predictions result
        preds = np.zeros(X_test.shape[1]) # number of testing observations
        
        # n = self.X_train.shape[1] # number of training observations
        print(self.X_train.shape[1], type(self.X_train.shape[1]))
        
        # reshape test sample Y
        y_s = np.transpose(X_test) # y_s = nxp, y_s[i] = (p,)
        
        # find alphas hat and beta hat
        for i in range(y_s.shape[0]):
            # xL2 = np.random.rand((self.X.shape[1]))
            
            # print((np.linalg.pinv(self.X_train).shape), y_s[i].shape)
            alphas_0 = np.linalg.pinv(self.X_train) @ y_s[i]
            betas_0 = (np.linalg.pinv(self.X_train) ** 2) @ y_s[i]
            x0 = np.concatenate((alphas_0, betas_0), axis=0)
            # print(x0)
            reg = poly_regression(self.X_train, y_s[i], self.group_vect, self.lambda_)
            coefs = reg.lasso_optimize(x0).x
            # print(reg.lasso_optimize(xL2).message)
            # print(coefs)
            poly_EkNN_R = single_polynomial_EkNN_R(self.X_train, y_s[i], coefs[0:self.X_train.shape[1]], coefs[self.X_train.shape[1]:], self.y_train, self.k)
            preds[i] = poly_EkNN_R.classify()

        return preds
        
        
        
    def score(self, Y, y_labels):
        """
        Calculate the accuracy of the classifier.
        """
        y_pred = self.predict(Y)
        return accuracy_score(y_labels, y_pred)  
            