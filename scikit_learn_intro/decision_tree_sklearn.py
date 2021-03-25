import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from scikit_learn_intro.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=1, stratify = y)

tree_model = DecisionTreeClassifier(criterion = "gini", max_depth= 4, random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plt = plot_decision_regions(X_combined,
                            y_combined,
                            classifier=tree_model,
                            test_idx = range(105, 150))

plt.show()

plot_tree(tree_model)

forest = RandomForestClassifier(criterion = "gini",
                                n_estimators=50,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)

plt = plot_decision_regions(X_combined,
                            y_combined,
                            classifier=forest,
                            test_idx = range(105, 150))


y_pred_tree = tree_model.predict(X_test)
y_pred_forest = forest.predict(X_test)

print("Accuracy for decision tree is {:.8f}".format(accuracy_score(y_test, y_pred_tree)))
print("Accuracy for random forest is {:.8f}".format(accuracy_score(y_test, y_pred_forest)))

