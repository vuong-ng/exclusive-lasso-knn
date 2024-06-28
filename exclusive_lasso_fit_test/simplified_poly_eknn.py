import numpy as np
import pandas as pd
import os # not used so far
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
from exclusive_lasso_opt import coordinateDescentFit


class exclusive_lasso:
    def __init__(self, X, y,  groups_vect, lambda_):
        self.X = X # predictors pxn matrix
        self.y = y # test sample pxn_test
        self.groups_vect = groups_vect
        self.lambda_ = lambda_ # cross_validate then pass into the function


    def regularized_least_square(self,alphas):
        """calculate the residual of X*/alphas compared to the test sample y

        Args:
            alphas (nx1 vector): coefficients vector

        Returns:
            integer: residual between X*/alphas and y
        """
        residual = (self.y**2) - ((self.X**2) @ alphas)
        # print(self.groups_vect.shape, "\n", alphas.shape)
        grouped_alphas = self.groups_vect @ np.absolute((alphas)) 
        return (np.sum(residual ** 2) + self.lambda_ * (np.sum(grouped_alphas ** 2))) # Square root of residuals plus the lasso penalty

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
        self.coefs_vect = coefs_vect # result from lasso_optimize function
        self.k = k
        self.x_labels = x_labels
        self.X = X # predict = pxn matrix
        
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
            class_coefs.append(self.class_coefs_vect(i))
        
        coefs_sums = [np.sum(class_coefs[i]) for i in range(len(class_coefs))]
        largest_coefs = max(coefs_sums)
        
        return coefs_sums.index(largest_coefs)
            
        
        #calculate L1-norm of coeficient vector
        
class EkNN_R:
    def __init__(self, X, y, coefs_vect, x_labels, k):
        self.coefs_vect = coefs_vect
        self.k = k
        self.x_labels = x_labels # mxn matrix that represent groups of each observation by 0,1
        self.X = X # predictors matrix pxn 
        self.y = y # test sample matrix pxn_test
        
        # construct nx1 vector containing k non-zero coeficients
        # which are k largest coefficients
        # Other coefficients that are not one of the k largest remains 0
        coefs = self.coefs_vect
        coefs = coefs.tolist()
        coefs.sort(reverse=True)
        max_coefs = coefs[:self.k]
        k_largest_coefs = self.coefs_vect

        for j in range(len(self.coefs_vect)):
            if k_largest_coefs[j] not in max_coefs:
                k_largest_coefs[j] = 0
                
        self.k_largest_coefs_vect  = k_largest_coefs
    
    def class_coefs_vect(self, label_index):
        """represent coeficients according to a particular class labels 

        Args:
            label_index (_type_): _description_
            k_largest_coefs (_type_): vector containing only k largest optimal coeficients
        """
        
        c_coefs = np.zeros(self.X.shape[1])
        for i in range(self.X.shape[1]): # columns index = index of observations
            if self.x_labels[label_index,i] == 1:
                c_coefs[i] = self.k_largest_coefs_vect[i]
            else:
                c_coefs[i] = 0
        return c_coefs # this can just be a list
    

    def create_obs_vects(self):
        """helper function to generate vector for each training data (each observation)
            to calculate the distance later.

        Returns:
            _list_: list containing n nx1 vectors representing n observations in the 
            training set
        """
        return np.transpose(self.X)

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
        d_vect = np.zeros(self.X.shape[1]) # number of observations
        
        for i in range(self.X.shape[1]):
            if self.k_largest_coefs_vect[i] == 0:
                d_vect[i] = 0 # is it safe to set those not k largest to 0
            else: 
                d_vect[i] = np.linalg.norm(self.y - self.k_largest_coefs_vect[i] * x_vects[i], ord=2)
                
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
            class_coefs_mat[i,:] = (self.class_coefs_vect(i))
        
        sum_coefs_weights = class_coefs_mat @ W
        # return W
        return np.where(sum_coefs_weights==max(sum_coefs_weights))[0][0]

class ExclusiveLassoKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_=1.0, group_num=1, k=3, step_size=0.01):
        self.lambda_ = lambda_
        self.group_num = group_num
        self.k = k
        self.step_size = step_size
    
    def fit(self, X, x_labels):
        
        # X passed in have nxp dimension
        self.group_vect = self.group_encode(X) 
        # print ("here: ",self.group_vect.shape)
        self.X = np.transpose(X) # X_train
        # self.x_labels = x_labels # y_train 
    
        class_names = [i for i in range(len(np.unique(x_labels)))]
        
        self.x_labels = self.label_matrix(class_names, x_labels)

        self.is_fitted_ = True

        return self
    
    def predict(self, Y):
        # Y test sample
        preds = np.zeros(Y.shape[1])
        
        # modify y's shape
        y_s = np.transpose(Y)
        # print(self.X.shape)
        for i in range(y_s.shape[0]):
            # xL2 = np.random.rand((self.X.shape[1]))
            xL2 = np.linalg.pinv(self.X) @ y_s[i]
            print(self.group_vect)
            exclusive_lasso_reg = coordinateDescentFit(self.lambda_, self.step_size, xL2, self.X, y_s[i], self.group_vect )
            coefs = exclusive_lasso_reg.coord_descent()
            # print(reg.lasso_optimize(xL2).message)
            knn_R = EkNN_R(self.X, y_s[i], coefs, self.x_labels, self.k)
            preds[i]=(knn_R.predict())

        return preds
    
    
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
    
    
    def group_encode(self, X):
        # call before having X
        # n = X.shape[1]
        # group_pop = int(n/self.group_num)
        # group_vect = []
        # start = 0
        # end = group_pop
        # while end <= n:
        #     temp = np.zeros(X.shape[1])
        #     if end + group_pop > n:
        #         temp[start:n] = 1
        #         group_vect.append(temp)
        #         break
        #     else: temp[start:end] = 1
        #     start = end
        #     end += group_pop
        #     group_vect.append(temp)
        # return np.array(group_vect)
    
        # 2nd approach perform unsupervised learning
        kmeans = KMeans(n_clusters=self.group_num, random_state=0, n_init="auto").fit(X)
        result = kmeans.labels_
        group_vect = np.zeros((self.group_num, len(result)))
        for i in range(self.group_num):
            for j in range(len(result)):
                if result[j] == i:
                    # print(result[j],"and ",i )
                    group_vect[i,j] = 1
        # print (group_vect.shape)
        return group_vect



    def score(self, Y, y_labels):
        """
        Calculate the accuracy of the classifier.
        """
        y_pred = self.predict(Y)
        return accuracy_score(y_labels, y_pred)  
            
        
        
        