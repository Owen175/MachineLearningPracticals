from sklearn import linear_model
import _pickle as cp
import numpy as np

# Given setup
X, y = cp.load(open('winequality-white.pickle', 'rb'))
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train
X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]


# Linear model like in p1. 
# reg = linear_model.LinearRegression()
# reg.fit(X_train, y_train)
# x = reg.predict(X_test) - y_test
# x = np.square(x)
# x = np.mean(x)
# print(x)


# Ridge minimises ||Xw-y|| + alpha ||w||. Sklearn will solve it automatically using an efficient algorithm for your usecase
# Can be useful to tune alpha. 
for i in range(5):
    reg = linear_model.Ridge(alpha=10**(i-2))
    reg.fit(X_train, y_train)
    x = reg.predict(X_test) - y_test
    x = np.square(x)
    x = np.mean(x)
    print("RIDGE: MSE for alpha =",10**(i-2), "is:", x)

