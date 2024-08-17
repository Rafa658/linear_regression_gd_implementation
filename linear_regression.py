import numpy as np

class LinearRegression:
    # y = m1x1 + m2x2 + ... + mnxn + c

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0

    def fit(self, X, y, epochs = 1000, learning_rate = 0.00001, tol = 1e-3):

        n_samples = X.shape[0]

        X = np.c_[np.ones(n_samples), X]

        self.coef_ = np.zeros(X.shape[1])

        mse_history = []
        for i in range(epochs):
            y_pred = X.dot(self.coef_)
                        
            gradients = 2/n_samples * X.T.dot(y_pred - y)

            mse = np.mean((y - y_pred) ** 2)
            mse_history.append(mse)

            self.coef_ -= learning_rate * gradients        

            if len(mse_history) <= 1:
                print(f'MSE = {mse}')
            else:
                print(f'MSE = {mse}, deltaMSE = {mse_history[-2] - mse_history[-1]}')

            if i > 0 and abs(mse_history[-2] - mse_history[-1]) < tol:
                break

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]  

        return mse_history 

    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_
    
    def score(self, X, y):
        y_pred = self.predict(X)

        u = sum((y - y_pred) ** 2)
        v = sum((y - y.mean()) ** 2)

        return np.sqrt(1 - u/v)