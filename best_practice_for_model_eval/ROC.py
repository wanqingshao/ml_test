from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = None)

X = df.iloc[:,2:].values
y = df.iloc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify=y, random_state= 1)

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", C = 100))

X_train2 = X_train[:, [4,14]] # only use two features

cv = list(StratifiedKFold(n_splits=3, random_state=1, shuffle = True).split(X_train2, y_train))

mean_tpr = 0
mean_fpr = np.linspace(0,1, 100)

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:,1], pos_label= 1)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = "ROC fold {} (area = {:.2f})".format(i+1, roc_auc))


## Fit with all data

probas_all_train = pipe_lr.fit(X_train, y_train).predict_proba(X_train)
probas_all_test = pipe_lr.fit(X_test, y_test).predict_proba(X_test)

all_train_fpr, all_train_tpr, all_train_thresholds = roc_curve(y_train, probas_all_train[:,1], pos_label= 1)
all_test_fpr, all_test_tpr, all_test_thresholds = roc_curve(y_test, probas_all_test[:,1], pos_label= 1)

all_train_auc = auc(all_train_fpr, all_train_tpr)
all_test_auc = auc(all_test_fpr, all_test_tpr)

plt.plot(all_train_fpr, all_train_tpr, label = "All train (area = {:.2f})".format(all_train_auc))
plt.plot(all_test_fpr, all_test_tpr, label = "All test (area = {:.2f})".format(all_test_auc))


plt.plot([0,1], [0,1], linestyle = "--", color = (0.5, 0.5, 0.5), label = "Random guessing")
plt.plot([0,0,1], [0,1,1], linestyle = ":", color = "black", label = "Perfect performance")
plt.xlabel("FPR")
plt.ylabel("TPR")

plt.legend()