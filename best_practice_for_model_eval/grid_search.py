from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import  make_pipeline
import pandas as pd


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = None)

X = df.iloc[:,2:].values
y = df.iloc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify=y, random_state= 1)

pipe_lr = make_pipeline(StandardScaler(),
                        SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = [{"svc__C": param_range,
               "svc__kernel": ["linear"]},
              {"svc__C": param_range,
               "svc__gamma" : param_range,
               "svc__kernel": ["rbf"]}]

gs = GridSearchCV(estimator= pipe_lr, param_grid = param_grid,
                  scoring="accuracy", cv = 10, refit = True, n_jobs=2)

gs = gs.fit(X_train, y_train)