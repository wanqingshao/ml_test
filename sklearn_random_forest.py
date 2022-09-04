from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=1, stratify=y)

dt = DecisionTreeClassifier(criterion= "gini", random_state=1, max_depth= 3)
rf = RandomForestClassifier(criterion="gini", n_estimators= 25, random_state=1, max_depth=3)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_hat_dt = dt.predict(X_test)
y_hat_rf = rf.predict(X_test)

acc_score_dt = accuracy_score(y_test, y_hat_dt)
f1_score_dt = f1_score(y_test, y_hat_dt, average = "weighted")

acc_score_rf = accuracy_score(y_test, y_hat_rf)
f1_score_rf = f1_score(y_test, y_hat_rf, average = "weighted")

print("Decision Tree: Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score_dt, f1_score_dt))
print("Random Forest: Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score_rf, f1_score_rf)) ## random forest and decision tree have similar performance in the current setting

tree.plot_tree(dt)
plt.show()