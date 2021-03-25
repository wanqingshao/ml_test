from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## Sequential backward selection

class SBS():
    def __init__(self, estimator, k_features, scoring = accuracy_score,
                 test_size = 0.25, random_state = 1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def _cal_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test, y_pred)
        return score


    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state= self.random_state,
                                                            stratify=y)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self._cal_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r = dim - 1):
                score = self._cal_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self




import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values , df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


knn = KNeighborsClassifier(n_neighbors= 5)
sbs = SBS(knn, k_features = 1)
sbs.fit(X_train_std, y_train)
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker = "o")
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.show()

min_feature = sbs.subsets_[np.max(np.where(np.array(sbs.scores_) == 1))]

knn.fit(X_train_std, y_train)
print("Accuracy of trainning group -- original data: {:5f}".format(knn.score(X_train_std, y_train)))
print("Accuracy of test group -- original data: {:5f}".format(knn.score(X_test_std, y_test)))


knn.fit(X_train_std[:, list(min_feature)], y_train)
print("Accuracy of trainning group -- after feature selection: {:5f}".format(knn.score(X_train_std[:, list(min_feature)], y_train)))
print("Accuracy of test group -- after feature selection: {:5f}".format(knn.score(X_test_std[:, list(min_feature)], y_test)))


## Random forest


from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators= 500,
                                random_state=1)
forest.fit(X_train, y_train)
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]

plt.bar(range(X_train.shape[1]),
        importance[indices],
        align = "center")
plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation = 90)
plt.tight_layout()

