import os
import pandas as pd
import numpy as np
from perceptron import plotting
from perceptron import perceptron
from perceptron import adaline
from perceptron import adalineSGD
import matplotlib.pyplot as plt

s = os.path.join("https://archive.ics.uci.edu", "ml", "machine-learning-databases","iris", "iris.data")

df = pd.read_csv(s, header = None, encoding = "utf-8")
y=df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values
X_std = X.copy()
X_std[:,0] = (X_std[:,0] - X_std[:, 0].mean()) / X[:, 0].std()
X_std[:,1] = (X_std[:,1] - X_std[:, 1].mean()) / X[:, 1].std()


plt.scatter(X[:50, 0], X[:50, 1], 1, color = "red", marker = "o", label = "setosa")
plt.scatter(X[50:, 0], X[50:, 1], 1, color = "blue", marker = "x", label = "versicolor")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc = "upper left")
plt.show()

ppn = perceptron.Perceptron(eta = 0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = "o")
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.legend(loc = "upper left")
plt.show()


ada1 = adaline.AdalineGD(n_iter=15, eta=0.01).fit(X,y)
ada2 = adaline.AdalineGD(n_iter=15, eta=0.0001).fit(X,y)


ada3 = adaline.AdalineGD(n_iter=15, eta=0.0001).fit(X_std,y)


fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15, 4))

ax[0].plot(range(1, len(ada1.cost_) +1),
           np.log10(ada1.cost_), marker = "o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log10(SSE)")
ax[0].set_title("Adaline -- learning rate 0.01")

ax[1].plot(range(1, len(ada2.cost_) +1),
           np.log10(ada2.cost_), marker = "o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log10(SSE)")
ax[1].set_title("Adaline -- learning rate 0.0001")

ax[2].plot(range(1, len(ada3.cost_) +1),
           np.log10(ada3.cost_), marker = "o")
ax[2].set_xlabel("Epochs")
ax[2].set_ylabel("log10(SSE)")
ax[2].set_title("Adaline -- learning rate 0.01 with standardized features")
plt.show()

ada_gd = adaline.AdalineGD(n_iter=15, eta = 0.01)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)


ada_sgd = adalineSGD.AdalineSGD(n_iter=15, eta = 0.01, random_state=1)
ada_sgd.fit(X_std, y)
plotting.plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title("Adaline == Stochastic Gradient Descient")
plt.xlabel("sepal length (zscore)")
plt.ylabel("petal length (zscore)")
plt.legend(loc = "upper left")

plt.show()

plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
