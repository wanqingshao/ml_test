import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from scikit_learn_intro import plotting


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
sc = StandardScaler()
sc.fit(X_train) # find mean and standard deviation using only the training data, this helps to prevent the information from the test data leaking into the training set
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("Accuracy {:.3f}".format(accuracy_score(y_test, y_pred)))

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plt = plotting.plot_decision_regions(X = X_combined_std, y = y_combined,
                               classifier=ppn, test_idx=range(X_train_std.shape[0], X_combined_std.shape[0]))
plt.xlabel("petal length (standardized)")
plt.ylabel("petal width (standardized)")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

