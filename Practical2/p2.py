from nbc import NBC
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


iris = load_iris()
X, y = iris['data'], iris['target']

# np.random.seed(42)
N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train

n_iterations = 200

nbc_accuracies = np.zeros(10)
lr_accuracies = np.zeros(10)

for i in range(n_iterations):
    # Shuffle data and outputs
    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    Xtrain = X[:N_train]
    ytrain = y[:N_train]
    Xtest = X[N_train:]
    ytest = y[N_train:]

    for p in range(1, 11):  # Train on 10p percent
        cut_Xtrain = Xtrain[:int(N_train*p/10)]
        cut_ytrain = y[:int(N_train*p/10)]

        nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
        nbc.fit(cut_Xtrain, cut_ytrain)
        lr = LR()
        lr.fit(cut_Xtrain, cut_ytrain)

        nbc_yhat = nbc.predict(Xtest)
        nbc_accuracies[p-1] += np.mean(nbc_yhat == ytest)
        lr_yhat = lr.predict(Xtest)
        lr_accuracies[p-1] += np.mean(lr_yhat == ytest)

# Plot the data.
plt.plot(range(10,101,10), nbc_accuracies/n_iterations, label="nbc", color="blue", marker="o")
plt.plot(range(10,101,10), lr_accuracies/n_iterations, label="lr", color="red", linestyle="--", marker="s")
plt.title("Comparison of LR and NBC")
plt.xlabel("Percentage of training data used")
plt.ylabel("Averaged Accuracy")
plt.legend()
plt.grid(True)
plt.show()
