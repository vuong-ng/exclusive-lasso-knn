import numpy as np

class coordinateDescentFit:

    def __init__(self, lambda_, step_size, alphas_0, X, y, group_matrix):
        self.lambda_ = lambda_
        self.step_size = step_size
        self.alphas_0 = alphas_0
        self.X = X # pxn matrix
        self.y = y # px1 test sample
        self.group_matrix = group_matrix
 
    def threshold(self, x1, x2): # S(x, lambda_) = sign(x)(|x| - lambda_)
        sgn = -1 if x1 < 0 else 1
        res = (abs(x1) - x2) if (abs(x1) - x2) > 0 else 0
        return sgn * res 

    def coord_descent(self):
        prev_alphas = np.zeros(self.X.shape[1])
        alphas = self.alphas_0
        # print("very first alpha: ", alphas[0])

        n = self.X.shape[1] # number of observations
        X_T = np.transpose(self.X) # nxp matrix

        while np.linalg.norm(alphas - prev_alphas, ord=1) > self.step_size:
        # for i in range()    
            # print(self.group_matrix)
            prev_alphas = alphas.copy()
            # print("initial alpha is: ",prev_alphas)
            for j in range(n): # update each of n coefficients sequentially
                if j == n-1:
                    X_l = self.X[:,:j]
                    alphas_l = alphas[:j]
                else:
                    X_l = np.concatenate((self.X[:,0:j], self.X[:,j+1:]), axis=1)
                    alphas_l = np.concatenate((alphas[:j], alphas[j+1:]))
                tilde_z = X_T[j] @ (self.y - X_l @ alphas_l)

                tilde_lambda_ = 0
                for g in range(len(self.group_matrix)):
                    if self.group_matrix[g,j] == 1: # find group of observation j
                        g_j = g
                        break
                for i in range(n):
                    if self.group_matrix[g_j,i] == 1 and i != j:
                        tilde_lambda_ += abs(alphas[i])


                tilde_lambda_ *= self.lambda_ 
                # print("tilde lambda: ", tilde_lambda_)
                x1 = tilde_z / (X_T[j] @ np.transpose(X_T[j]) + self.lambda_)
                # print("x1 is: ", x1)
                x2 = tilde_lambda_ / (X_T[j] @ np.transpose(X_T[j]) + self.lambda_)
                alphas_j = self.threshold(x1, x2)
                alphas[j] = alphas_j
                print(prev_alphas[j], alphas[j])
            # print("l1 norm of difference: ", np.linalg.norm(alphas - prev_alphas, ord=1))
            
        return alphas