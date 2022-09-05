from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=1, stratify=y)

lgr = LogisticRegression(C = 100, random_state=1, solver = "liblinear", multi_class="ovr")
lgr.fit(X_train, y_train)
y_hat = lgr.predict(X_test)

acc_score = accuracy_score(y_test, y_hat)
f_score = f1_score(y_test, y_hat, average = "weighted")


print("Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score, f_score)) ### much better than perceptron
