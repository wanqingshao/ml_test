from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, [1,2]]
y = iris.target
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lg = make_pipeline(StandardScaler(),
                   OneVsRestClassifier(LogisticRegression(penalty="l2", multi_class="ovr", solver = "lbfgs", random_state= 1)))
knn = make_pipeline(StandardScaler(),
                    OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 3)))

dt = OneVsRestClassifier(DecisionTreeClassifier(max_depth = 5))

mv = OneVsRestClassifier(VotingClassifier(estimators=[("lg", lg), ("knn", knn), ("dt", dt)], voting= "soft")) ## soft votting using predicted probability, hard votting uses predicted label

cf_dic = {"lg":lg, "knn":knn, "dt":dt, "mv":mv}
color_dic = {"lg":"green", "knn":"blue", "dt":"orange", "mv":"black"}

for cf_n, cf in cf_dic.items():
    cf.fit(X_train, y_train)
    y_score = cf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel()) ## this is micro averaging
    plt.plot(fpr, tpr, linestyle = "--", color = color_dic[cf_n], label = cf_n)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc = "lower right")
plt.show()