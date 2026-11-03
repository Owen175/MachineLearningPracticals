import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt

# Given setup
X, y = cp.load(open('winequality-white.pickle', 'rb'))
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

def makeBarChart():
    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar(unique, counts)
    plt.title("training outcomes distribution")
    plt.show()

class MostTrivialPredictor:
    def __init__(self):
        unique, counts = np.unique(y_train, return_counts=True)
        self.avg_y = np.dot(unique, counts)/N

    def predict(self, _):
        return self.avg_y
    
    def getTestMSE(self):
        ys = np.array([self.predict(xi) for xi in X_test])
        ys = ys - y_test
        ys = np.square(ys)
        return np.mean(ys)

class LinearPredictor:
    def __init__(self, n=N_train):
        self.means, self.sds, X_s = self.standardise(n)
        self.ws = (np.linalg.inv(np.transpose(X_s) @ X_s) @ (np.transpose(X_s))).dot(y_train[:n]) 
        self.X_train_standardised = X_s
        self.nTraining = n

    def standardise(self, n):
        # Standardises each of the 11 features.
        Xt = X_train[:n]
        means = np.apply_along_axis(np.mean, 0, Xt)
        sds = np.sqrt(np.apply_along_axis(np.var, 0, Xt))
        # Need to subtract means, scale by 1/vars, add col of 0s to allow for w0 to be found
        return means, sds, np.c_[np.ones(n), (Xt-means)/sds]

    def predict(self, xi):
        xi_s = xi - self.means
        xi_s = xi_s/self.sds
        xi_s = np.insert(xi_s, 0, 1)
        return np.dot(self.ws, xi_s)

    def getTestMSE(self):
        X_test_s = X_test - self.means
        X_test_s = X_test_s/self.sds
        X_test_s = np.c_[np.ones(N_test), X_test_s]
        ys = np.dot(X_test_s, self.ws)
        ys = ys - y_test
        ys = np.square(ys)
        return np.mean(ys)

    def getTrainingMSE(self):
        X_train_s = self.X_train_standardised
        ys = np.dot(X_train_s, self.ws)
        ys = ys - y_train[:self.nTraining]
        ys = np.square(ys)
        return np.mean(ys)        


def makeComparisonChart():
    linPredictors = [LinearPredictor(i) for i in range(20, 600, 20)]
    training_error = [lp.getTrainingMSE() for lp in linPredictors]
    testing_error = [lp.getTestMSE() for lp in linPredictors]
    plt.plot(range(20, 600, 20), training_error, label="Training Error")
    plt.plot(range(20, 600, 20), testing_error, label="Testing Error")

    plt.title("training vs testing error distribution")
    plt.show()

# makeBarChart()

print("MSE for simplest predictor: ", MostTrivialPredictor().getTestMSE())

lp = LinearPredictor()
print("Training MSE for linear regression on full test set: ", lp.getTestMSE())
print("Training MSE for linear regression on full training set: ", lp.getTrainingMSE())

makeComparisonChart()
