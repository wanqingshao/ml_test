import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve, train_test_split, GridSearchCV
#from sklearn.metrics import accuracy_score  ### No need to use this, model.score function can be used for accuracy
from sklearn.metrics import make_scorer, f1_score


"""
Constructing a simple pipeline
"""
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = None)

le = LabelEncoder()


X = df.iloc[:, 2:].values
y = le.fit_transform(df.iloc[:,1].values)
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, test_size=0.3)

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", max_iter=10000))

pipe_lr.fit(X_train, y_train)

#y_hat = pipe_lr.predict(X_test)
#acc_score = accuracy_score(y_test, y_hat)

print("Accuracy score is {:.3f}".format(pipe_lr.score(X_test, y_test)))

"""
cross validation score
"""

cv_score = cross_val_score(estimator=pipe_lr, X = X_train, y = y_train, cv = 10 , n_jobs=2)
print("10 fold CV score is {:.3f} +/- {:.3f}".format(np.mean(cv_score), np.std(cv_score)))


"""
Learning curve
"""

pipe_lr2= make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2", random_state=1, solver="lbfgs", max_iter=10000))



train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr2, X= X_train, y = y_train,
                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                        cv = 10,
                                                        n_jobs=2) ## Function learning_curve determines cross-validated training and test scores for different training set sizes

train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)


plt.plot(train_sizes, train_mean,
         color = "blue", marker = "o",
         markersize = 5, label = "training accuracy")
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = "blue")

plt.plot(train_sizes, test_mean,
         color = "green", marker = "s",
         markersize = 5, label = "test accuracy",
         linestyle = "--")
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = "green")

plt.grid()
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.ylim(0.8,1.03)
plt.legend(loc = "lower right")
plt.show()


"""
Validation curve
"""

param_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, test_scores = validation_curve(estimator=pipe_lr2, X = X_train, y = y_train,
                                             param_name= "logisticregression__C", ## pipe_lr.get_params() will show parameter names
                                             param_range = param_ranges,
                                             cv = 10, n_jobs=2) # results are matrix nrow = n_param, ncol = cv.


train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

plt.plot(param_ranges, train_mean,
         color = "blue", marker = "o",
         markersize = 5, label = "training accuracy")
plt.fill_between(param_ranges, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = "blue")

plt.plot(param_ranges, test_mean,
         color = "green", marker = "s",
         markersize = 5, label = "test accuracy",
         linestyle = "--")
plt.fill_between(param_ranges, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = "green")

plt.grid()
plt.xlabel("Parameter C in LogisticRegression (log)")
plt.ylabel("Accuracy")
plt.xscale("log")
plt.ylim(0.8,1.03)
plt.legend(loc = "lower right")
plt.show()

"""
Grid search
"""

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_ranges = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

scorer = make_scorer(f1_score, pos_label = 0)


param_grid = [{"svc__C": param_ranges,
               "svc__kernel": ['linear']},
              {"svc__C": param_ranges,
               "svc__gamma": param_ranges,
               "svc__kernel": ['rbf']}]

gs = GridSearchCV(estimator= pipe_svc,
                  param_grid = param_grid,
                  scoring= scorer, ## can use customized scoring function, it just needs to be score_func(y, y_pred, **kwargs)
                  cv = 10,
                  refit = True, ## refit: refit  an estimator using the best found parameters
                  n_jobs= -1) ## n_jobs = -1 means using all processors

## Grid search is computationally expensive, an alternative is RandomizedSearchCV, usually performs as well as grid search
gs.fit(X_train, y_train)

print(gs.best_params_)