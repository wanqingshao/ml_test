import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RANSACRegressor, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch10/housing.data.txt', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#sns.pairplot(df)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=1)

def get_performance(y_test, y_hat, alg):
    r2_score_r = r2_score(y_test, y_hat)
    mse_r = mean_squared_error(y_test, y_hat)
    print(alg + ": R2 score is {:.2f}, MSE is {:.2f}".format(r2_score_r, mse_r))
    return None


"""Simple linear model"""

params = [{'sgdregressor__alpha': [0.0001, 0.001, 0.1,1, 10, 100],
          'sgdregressor__penalty': ['l2', 'l1']},
          {'sgdregressor__alpha': [0.0001, 0.001, 0.1, 1, 10, 100],
           'sgdregressor__penalty': ['elasticnet'],
           "sgdregressor__l1_ratio": np.linspace(0,1, num = 21)}]


gsd_pipe = make_pipeline(StandardScaler(),  SGDRegressor(random_state = 1))

lr_gs = GridSearchCV(estimator = gsd_pipe, param_grid=params)
lr_gs.fit(X_train, y_train)
print(lr_gs.best_params_)
y_hat_lr = lr_gs.predict(X_test)
get_performance(y_test, y_hat_lr, "Linear regression with Grid Search")


"""RANSAC"""

ransac_r = make_pipeline(StandardScaler(), RANSACRegressor(random_state = 1))
ransac_r.fit(X_train, y_train)
y_hat_ransac = ransac_r.predict(X_test)
get_performance(y_test, y_hat_ransac, "RANSAC") ## worse than grid searched SGD


"""polynomial"""

p_pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), SGDRegressor(random_state=1))

params2 = [{"polynomialfeatures__degree":[2,3,4],
            "sgdregressor__penalty": ["elasticnet"],
            "sgdregressor__l1_ratio": np.linspace(0,1, num = 21),
            "sgdregressor__alpha": [0.0001, 0.001, 0.1, 1, 10, 100]}]

poly_gs = GridSearchCV(estimator=p_pipe, param_grid=params2)
poly_gs.fit(X_train, y_train)
print(poly_gs.best_params_)
y_hat_poly = poly_gs.predict(X_test)
get_performance(y_test, y_hat_poly, "Polynomial with grid search")


"""random forest"""

rf = RandomForestRegressor(random_state=1)
rf.fit(X_train, y_train)
y_hat_rf = rf.predict(X_test)
get_performance(y_test, y_hat_rf, "Random forest regressor") ## works the best
