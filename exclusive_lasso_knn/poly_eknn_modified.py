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

class polynomial_reg:
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
    
class polynomial_EkNN_R:
    def __init__(self, X, y, alphas, betas, x_labels_encode, k):
        # self.alphas = alphas 
        # self.betas = betas
        self.k = k
        self.X = X #pxn matrix
        self.y = y # px1 vector
        self.x_labels_encode = x_labels_encode 
    
        sum_coefs = alphas + betas
        sum_coefs_list = sum_coefs.tolist()
        sum_coefs_list.sort(reverse=True)
        print(sum_coefs_list)
        max_sum_coefs_list = sum_coefs_list[0:self.k]
        print(sum_coefs_list)
        
        # alphas_0 = np.linalg.pinv(X) @ y
        # betas_0 = (np.linalg.pinv(X) ** 2) @ y
        # x0 = np.concatenate((alphas_0, betas_0))
        # coefficients = 
        
        k_largest_sum_coefs = alphas + betas
        
        for j in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[j] not in max_sum_coefs_list:
                k_largest_sum_coefs[j] = 0
        
        # for i in range(len(sum_coefs_list)):
        #     if k_largest_sum_coefs[i] == 0:
                # alphas[i] = 0
                # betas[i] = 0
        
        # diagonal matrix with alphas and betas on the diagonal
        self.alphas_diag_matrix = np.diag(alphas)
        self.betas_diag_matrix = np.diag(betas)
        # self.largest_alphas = alphas
        # self.largest_betas = betas
        
        # nx1 vector to calculate the label ly = w . (alpha+beta) 
        self.k_largest_sum_vect = k_largest_sum_coefs
        
    def class_coefs_matrix(self):

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
        class_weight_matrix = class_coefs_matrix @ np.diag(w) 
        res = 0
        max_so_far = 0
        for i in range(class_coefs_matrix.shape[0]):
            if np.amax(class_weight_matrix[i,:]) > max_so_far:
                max_so_far = np.amax(class_weight_matrix[i,:])
                res = i
        return res
            
            
        
        
        
        
# class polynomial_EkNN_R_classifier:
#     def __init__(self, lambda_=1.0,  group_num=1, k=3):
#         self.lambda_ = k
#         self.group_num = group_num
#         self.k = k
        
#     def fit(self, X_train, y_train):
#         self.group_vect = self.group_encode(X_train)
#         self.X_train = np.transpose(X_train)
#         class_names = [i for i in range(len(np.unique(y_train)))]
            
#         #generate label for y
#         self.y_train = self.label_matrix(class_names, y_train)
#         self.is_fitted_ = True
#         # return self

#         #generate alphas and betas
#         coefficients = polynomial_reg(self.X_train, )
        
#     def 