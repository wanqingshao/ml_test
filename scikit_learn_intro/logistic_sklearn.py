import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scikit_learn_intro.plotting import  plot_decision_regions
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
sc = StandardScaler()
sc.fit(X_train) # find mean and standard deviation using only the training data, this helps to prevent the information from the test data leaking into the training set
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))



lr = LogisticRegression( C  = 100, random_state=1,solver = "lbfgs", multi_class = "multinomial")
lr.fit(X_train_std, y_train)
plt = plot_decision_regions(X_combined_std,
                            y_combined,
                            classifier=lr,
                            test_idx = range(X_train_std.shape[0], X_combined_std.shape[0]))
plt.xlabel("petal length (zscore)")
plt.ylabel("petal width (szcore")
plt.legend(loc = "upper left")
plt.show()