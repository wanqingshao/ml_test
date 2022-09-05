import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import f1_score, accuracy_score



df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

df.columns = ["label", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
              "Flavanoids","Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
              "OD280/OD315 of diluted wines", "Proline"]

y = df["label"].values
X = df.iloc[:,1:].values

X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=1, stratify= y, shuffle=True)

## Train with original features

lg = LogisticRegression(penalty = "l2", C = 1, class_weight= "balanced", random_state=1,
                        solver = "liblinear", multi_class = "ovr")
lg.fit(X_train, y_train)
y_hat = lg.predict(X_test)

acc_score = accuracy_score(y_test, y_hat)
f_score = f1_score(y_test, y_hat, average="weighted")


## Train with PCA constructed features

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test) ## no need to fit again

lg2 = LogisticRegression(penalty = "l2", C = 1, class_weight= "balanced", random_state=1,
                        solver = "liblinear", multi_class = "ovr")
lg2.fit(X_train_pca, y_train)
y_hat2 = lg2.predict(X_test_pca)

acc_score2 = accuracy_score(y_test, y_hat2)
f_score2 = f1_score(y_test, y_hat2, average="weighted")

## Train with LDA constructed features

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train) ## supervised algorithm, y_train needed
X_test_lda = lda.transform(X_test) ## no need to fit again, transform without label

lg3 = LogisticRegression(penalty = "l2", C = 1, class_weight= "balanced", random_state=1,
                        solver = "liblinear", multi_class = "ovr")
lg3.fit(X_train_lda, y_train)
y_hat3 = lg3.predict(X_test_lda)

acc_score3 = accuracy_score(y_test, y_hat3)
f_score3 = f1_score(y_test, y_hat3, average="weighted")

print("Original: Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score, f_score))
print("PCA feature: Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score2, f_score2))
print("LDA feature: Accuracy score is {:.3f}, F1 score is {:.3f}".format(acc_score3, f_score3))
