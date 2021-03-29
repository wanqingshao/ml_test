from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = np.exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]

    # Collect the top k eigenvectors (projected samples)
    X_pc = eigvecs[:,range(n_components)]

    return np.copy(X_pc)


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0,0], X[y==0,1], color = "red", marker  = "x", alpha = 0.5)
plt.scatter(X[y==1,0], X[y==1,1], color = "blue", marker  = "o", alpha = 0.5)




from sklearn.decomposition import PCA
scikit_pca = PCA(n_components=2)
X_pca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))

ax[0].scatter(X_pca[y==0,0], X_pca[y==0,1], color = "red", marker  = "x", alpha = 0.5)
ax[0].scatter(X_pca[y==1,0], X_pca[y==1,1], color = "blue", marker  = "o", alpha = 0.5)
ax[0].set_xlabel( "PC1" )
ax[0].set_ylabel("PC2")
ax[0].set_title("simple PCA")


X_kpca = rbf_kernel_pca(X, gamma = 15, n_components=2)

ax[1].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color = "red", marker  = "x", alpha = 0.5)
ax[1].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color = "blue", marker  = "o", alpha = 0.5)
ax[1].set_xlabel( "PC1" )
ax[1].set_ylabel("PC2")
ax[1].set_title("RBF kernel PCA")


from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=123, noise = 0.1, factor = 0.2)
plt.scatter(X[y==0,0], X[y==0,1], color = "red", marker  = "x", alpha = 0.5)
plt.scatter(X[y==1,0], X[y==1,1], color = "blue", marker  = "o", alpha = 0.5)

scikit_pca = PCA(n_components=2)
X_pca = scikit_pca.fit_transform(X)
X_kpca = rbf_kernel_pca(X, gamma =15, n_components=2)

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))

ax[0].scatter(X_pca[y==0,0], X_pca[y==0,1], color = "red", marker  = "x", alpha = 0.5)
ax[0].scatter(X_pca[y==1,0], X_pca[y==1,1], color = "blue", marker  = "o", alpha = 0.5)
ax[0].set_xlabel( "PC1" )
ax[0].set_ylabel("PC2")
ax[0].set_title("simple PCA")

ax[1].scatter(X_kpca[y==0,0], X_kpca[y==0,1], color = "red", marker  = "x", alpha = 0.5)
ax[1].scatter(X_kpca[y==1,0], X_kpca[y==1,1], color = "blue", marker  = "o", alpha = 0.5)
ax[1].set_xlabel( "PC1" )
ax[1].set_ylabel("PC2")
ax[1].set_title("RBF kernel PCA")

## modified rbf_kernel_pca, turn both eigenvalues and eigenvectors

def rbf_kernel_pca_mod(X, gamma, n_components):
    dist = pdist(X, 'sqeuclidean')
    mat_dist = squareform(dist)
    K = np.exp(-gamma * mat_dist)
    n = K.shape[0]
    n_ones = np.ones((n,n)) /n
    K_centered = K - n_ones.dot(K) - K.dot(n_ones) + n_ones.dot(K).dot(n_ones)

    eigenvals, eigenvecs = eigh(K_centered)
    eigenvals, eigenvecs = eigenvals[::-1], eigenvecs[:,::-1]
    return np.copy(eigenvals[range(n_components)]), np.copy(eigenvecs[:,range(n_components)])


X, y = make_moons(n_samples= 100, random_state=123)
lambdas, alphas =  rbf_kernel_pca_mod(X, gamma = 15, n_components=2)

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

x_new = X[25]
x_proj = alphas[25]
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)


plt.scatter(alphas[y == 0, 0], alphas[y == 0, 1],color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], alphas[y == 1, 1],color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj[0], x_proj[1], color='black',label='original projection of point X[25]', marker='^', s=100)
plt.scatter(x_reproj[0], x_reproj[1], color='green',label='remapped point X[25]', marker='x', s=500)
plt.legend()


## sklearn kernel pca

from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()