import numpy as np


class Lab1(object):
    def solver(self, A, b):
       
        return np.linalg.inv(A).dot(b)

    def fitting(self, x, y):
        one=np.ones(len(x))
        X=np.column_stack((x,one))
        X_inverse=np.linalg.pinv(X)
        coeff=X_inverse.dot(y)
       
        return coeff

    def naive5(self, X, A, Y):
        S = np.zeros((X.shape[0],Y.shape[0]))
    
        for i in range(0,X.shape[0]):
            for j in range(0,Y.shape[0]):
                S[i,j]=((X[i].dot(A))).dot((Y[j]))
        return S
    def matrix5(self, X, A, Y):
        return (X.dot(A)).dot(Y.T)

    def naive6(self, X, A):

        ans=np.zeros(X.shape[0])
        for i in range(0,X.shape[0]):
            ans[i]=X[i].dot(A).dot(X[i])
        return ans
    def matrix6(self, X, A):
        ans=np.sum(np.multiply(X.dot(A),X) , axis=1)
        return ans
