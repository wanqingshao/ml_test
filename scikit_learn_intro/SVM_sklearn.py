import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikit_learn_intro.plotting import plot_decision_regions
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


svm = SVC(kernel="linear", C =1, random_state= 1)
svm.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx = range(105, 150))
plt.xlabel("petal length (standardized)")
plt.ylabel("petal width (standardized)")
plt.legend(loc = "upper left")
plt.show()


## Version 2

#from sklearn.linear_model import SGDClassifier
#ppn = SGDClassifier(loss = "perceptron")
#lr = SGDClassifier(loss = "log")
#svm = SGDClassifier(loss = "hinge")

## Kernel svm

svm_k = SVC(kernel="rbf", random_state= 1, gamma = 0.1, C = 10)
svm_k.fit(X_train_std, y_train)
plt = plot_decision_regions(X_combined_std,
                            y_combined,
                            classifier=svm_k,
                            test_idx = range(105, 150))
plt.xlabel("petal length (standardized)")
plt.ylabel("petal width (standardized)")
plt.legend(loc = "upper left")
plt.show()