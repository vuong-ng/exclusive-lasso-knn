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
    def __init__(self, X, y, groups_vect, lambda_1, lambda_2):
        self.X = X
        self.y = y
        self.groups_vect = groups_vect
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
    def regularized_least_square(self, coefficients):
        
        # coefficients vector is alphas stacked upon betas
        alphas = coefficients[0:self.X.shape[1]]
        betas = coefficients[self.X.shape[1]:]
        grouped_alphas = self.groups_vect @ np.absolute(alphas)
        grouped_betas = self.groups_vect @ np.absolute(betas)
        residual= self.y + self.y**2 - (self.X @ alphas + self.X**2 @ betas)
        return (np.sum(residual**2) + self.lambda_1 * (np.sum(grouped_alphas ** 2)) + self.lambda_2 * (np.sum(grouped_betas ** 2)) )
    
    def lasso_optimize(self, x0):
        func = lambda stacked_coefficients: self.regularized_least_square(stacked_coefficients)
        return minimize(func, x0, method='SLSQP')
    
class polynomial_EkNN_C:
    def __init__(self, X, alphas, betas, x_labels_encode, k):
        self.alphas = alphas 
        self.betas = betas
        self.k = k
        self.X = X
        self.x_labels_encode = x_labels_encode # labels encoding
        
        # generate k largest coefficients for the model
        sum_coefs = alphas + betas
        sum_coefs_list = sum_coefs.tolist()
        sum_coefs_list.sort(reverse=True)
        max_sum_coefs_list = sum_coefs_list[:self.k]
        
        k_largest_sum_coefs = alphas + betas
        
        for j in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[j] not in max_sum_coefs_list:
                k_largest_sum_coefs[j] = 0
        
        for i in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[i] == 0:
                alphas[i] = 0
                betas[i] = 0
        
        self.largest_alphas = alphas
        self.largest_betas = betas
        self.k_largest_sum_coefs = k_largest_sum_coefs
        
        # self.k_largest_sum_coefs = largest_betas + largest_alphas
        
        # self.largest_alphas = largest_alphas
        # self.largest_betas = largest_betas

    def class_coefs_vect(self):
        
        class_coefs_vect = np.zeros((self.x_labels_encode.shape[0],self.X.shape[1])) # m classes x n number of observations 
        
        for i in range(len(self.x_labels_encode)):
            for j in range(self.X.shape[1]):
                if self.x_labels_encode[i, j] == 1:
                    class_coefs_vect[i, j] = self.k_largest_sum_coefs[j]
                else:
                    class_coefs_vect[i, i] = 0
        return class_coefs_vect
        
        # class_alphas_vect = np.zeros((len(self.x_labels_encode.shape[0]),self.X.shape[1])) # number of observations 
        
        # for i in range(len(self.x_labels_encode)):
        #     for j in range(len(self.X.shape[1])):
        #         if self.x_labels_encode[i, j] == 1:
        #             class_alphas_vect[i, j] = self.largest_alphas[j]
        #         else:
        #             class_alphas_vect[i, i] = 0
                    
        # class_betas_vect = np.zeros((len(self.x_labels_encode.shape[0]), self.X.shape[1]))
        
        # for i in range(len(self.x_labels_encode)):
        #     for j in range(len(self.X.shape[1])):
        #         if self.x_labels_encode[i,j] == 1:
        #             class_betas_vect[i,j] =  self.largest_betas[j]
        #         else:
        #             class_betas_vect[i,j] = 0
        # return class_alphas_vect, class_betas_vect
    
    def classify(self):
        class_coefficients_sum = self.class_coefs_vect()
        
        # coefficients_sum = class_alphas_vect + class_betas_vect
        coefs_sums = [np.sum(class_coefficients_sum[i]) for i in range(len(class_coefficients_sum))]
        return coefs_sums.index(max(coefs_sums))

class polynomial_EkNN_R:
    def __init__(self, X, y, alphas, betas, x_labels_encode, k):
        self.alphas = alphas 
        self.betas = betas
        self.k = k
        self.X = X
        self.y = y
        self.x_labels_encode = x_labels_encode # labels encoding
        
        # generate k largest coefficients for the model
        sum_coefs = alphas + betas
        sum_coefs_list = sum_coefs.tolist()
        sum_coefs_list.sort(reverse=True)
        max_sum_coefs_list = sum_coefs_list[:self.k]
        
        k_largest_sum_coefs = alphas + betas
        
        for j in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[j] not in max_sum_coefs_list:
                k_largest_sum_coefs[j] = 0
        
        for i in range(len(sum_coefs_list)):
            if k_largest_sum_coefs[i] == 0:
                alphas[i] = 0
                betas[i] = 0
        
        self.largest_alphas = alphas
        self.largest_betas = betas
        self.k_largest_sum_coefs = k_largest_sum_coefs

    def class_coefs_vect(self):
        
        class_coefs_vect = np.zeros((len(self.x_labels_encode.shape[0]),self.X.shape[1])) # m classes x n number of observations 
        
        for i in range(len(self.x_labels_encode)):
            for j in range(len(self.X.shape[1])):
                if self.x_labels_encode[i, j] == 1:
                    class_coefs_vect[i, j] = self.k_largest_sum_coefs[j]
                else:
                    class_coefs_vect[i, i] = 0
                    
        # class_betas_vect = np.zeros((len(self.x_labels_encode.shape[0]), self.X.shape[1]))
        
        # for i in range(len(self.x_labels_encode)):
        #     for j in range(len(self.X.shape[1])):
        #         if self.x_labels_encode[i,j] == 1:
        #             class_betas_vect[i,j] =  self.largest_betas[j]
        #         else:
        #             class_betas_vect[i,j] = 0
        return class_coefs_vect               
                
    # def create_obs_vects(self):
    #     return np.transpose(self.X) # return nxp matrix

        
    def distance(self):
        x_vect = np.transpose(self.X)
        d_vect = np.zeros(self.X.shape[1]) # number of observations
        
        for i in range(self.X.shape[1]):
            if self.k_largest_sum_coefs[i] == 0:
                d_vect[i] = 0 # is it safe to set those not k largest to 0
            else: 
                d_vect[i] = np.linalg.norm(self.y + (self.y**2) - (self.alphas[i] * x_vect[i,:] + self.betas[i] * (x_vect[i,:] ** 2)), ord=2)
                
        return d_vect

    def weights(self):
        d_vect = self.distance()
        weights = np.zeros(d_vect.shape[0])
        
        for i in range(self.X.shape[1]):
            if self.k_largest_sum_coefs[i] != 0:  # not one of k obs with the largest coefficients
                weights[i] = 0
            elif np.amax(d_vect) != np.amin(d_vect):
                weights[i] = ((np.amax(d_vect) - d_vect[i])/(np.amax(d_vect)-np.amin(d_vect)))
            else:
                weights[i] = 1
        return weights
    
    def classify(self):
        weights_coefs_product = self.weights() * self.k_largest_sum_coefs
        return np.where(weights_coefs_product == max(weights_coefs_product))
            

class polynomial_EkNN:
    def __init__(self, lambda_1, lambda_2, group_num, k):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
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
        #sth
        kmeans= Kmeans(n_cluster = self.group_num, random_state=0, n_init="auto").fit(X)
        result = kmeans.labels_
        group_vect = np.zeros((self.group_num, len(result)))
        for i in range(self.group_num):
            for j in range(len(result)):
                if result[j] == i:
                    # print(result[j],"and ",i )
                    group_vect[i,j] = 1
        # print (group_vect.shape)
        return group_vect

        
        
    def label_matrix(self, label_names:list, labels):
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
        preds = np.zeros(X_test.shape(1)) # number of testing observations
        
        n = self.X_train.shape[1] # number of training observations
        
        # reshape test sample Y
        y_s = np.tranpose(X_test) # y_s = nxp, y_s[i] = (p,)
        
        # find alphas hat and beta hat
        for i in range(y_s.shape[0]):
            # xL2 = np.random.rand((self.X.shape[1]))
            alphas_0 = np.linalg.pinv(self.X) @ y_s[i]
            betas_0 = (np.linalg.pinv(self.X) ** 2) @ y_s[i]
            x0 = np.concatenate((alphas_0, betas_0), axis=0)
            reg = polynomial_reg(self.X_train, y_s[i], self.group_vect, self.lambda_1, self.lambda_2)
            coefs = reg.lasso_optimize(x0).x
            # print(reg.lasso_optimize(xL2).message)
            poly_EkNN_R = polynomial_EkNN_R(self.X_train, y_s[i], coefs[:n,:], coefs[n:,:], self.y_train, self.k)
            preds[i] = poly_EkNN_R.classify()

        return preds
        
        
        
        
        
        