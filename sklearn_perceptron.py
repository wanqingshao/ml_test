from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

X_normed = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normed, y, random_state=1, stratify=y, test_size=0.3)

percep = Perceptron(eta0 = 0.1, random_state=1)
percep.fit(X_train, y_train)
y_pred = percep.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred, average = "weighted")

print("Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score, f_score))




