import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scikit_learn_intro.logistic import  Logistic
from scikit_learn_intro.plotting import  plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
#X_std = X.copy()
#X_std[:,0] = (X_std[:,0] - X_std[:, 0].mean()) / X[:, 0].std()
#X_std[:,1] = (X_std[:,1] - X_std[:, 1].mean()) / X[:, 1].std()


X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = Logistic(eta = 0.05, n_iter = 1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)

plt = plot_decision_regions(X = X_train_01_subset, y = y_train_01_subset,
                      classifier=lrgd)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.show()