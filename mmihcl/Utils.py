"""
Functions for other tools
"""
import numpy as np

from scipy.linalg import svd
from sklearn.cross_decomposition import CCA


def svdDenoise(array, n_components=20):
    """
    Calculate the best rank-n_components approximation of array by singular value decomposition.

    Parameters:
    array (numpy.ndarray): The array to be approximated.
    n_components (int): Number of components of array to keep.

    Returns:
    svd_array (numpy.ndarray): The array that has been approximated.
    """
    if array.shape[1] < n_components:
        n_components = array.shape[1]

    U, s, _ = svd(array)

    svd_array = U[:, :n_components] @ np.diag(s[:n_components])

    return svd_array


def getCancor(X, Y, n_components=20, max_iter=1000):
    """
    Fit CCA and calculate the canonical correlations.

    Parameters:
    X (numpy.ndarray): First array to be fitted.
    Y (numpy.ndarray): Second array to be fitted.
    n_components (int): Number of components of array to keep.
    max_iter (int): Maximum number of iterations.

    Returns:
    cancor (numpy.ndarray): Vector of canonical components.
    cca (CCA): object CCA.
    """    
    cca = CCA(n_components=n_components, max_iter=max_iter)

    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    cancor = np.corrcoef(X_c, Y_c, rowvar=False).diagonal(offset=cca.n_components)

    return cancor, cca


def correlationDistance(array_1, array_2):
    """
    Calculate the pair-wise 1 - Pearson correlations between array_1 and array_2.

    Parameters:
    array_1 (numpy.ndarray): The first array.
    array_2 (numpy.ndarray): The second array.

    Returns:
    dist_mat (numpy.ndarray): The distance matrix between array_1 and array_2,
        where (i, j)-th entry is the Pearson correlation between i-th row of array_1 and j-th row of array_2.
    """
    array_1 = (array_1.T - np.mean(array_1, axis=1)).T
    array_2 = (array_2.T - np.mean(array_2, axis=1)).T

    array_1 = (array_1.T / np.sqrt(np.sum(array_1 ** 2, axis=1))).T
    array_2 = (array_2.T / np.sqrt(np.sum(array_2 ** 2, axis=1))).T

    dist_mat = 1 - array_1 @ array_2.T

    return dist_mat


def correctDistance(dist_mat, min_dist=1e-7):
    """
    Correct the distance matrix to make sure that min(distance matrix) >= minimum distance.

    Parameters:
    dist_mat (numpy.ndarray): The distance matrix.
    min_dist (float): The minimum distance.

    Returns:
    cor_dist_mat (numpy.ndarray): The corrected dist_mat.
    """
    cur_min_dist = np.min(dist_mat)

    if cur_min_dist < min_dist:
        cor_dist_mat = dist_mat - cur_min_dist + min_dist
    else:
        cor_dist_mat = dist_mat

    return cor_dist_mat


def aggregateAdjacentMatrix(mat):
    """
    Calculate the adjacent-aggregated version of a matrix.

    Parameters:
    mat (numpy.ndarray): The matrix to be aggregated.

    Returns:
    agg_mat (numpy.ndarray): The matrix that has been approximated.
    """
    mat += np.eye(mat.shape[0])
    
    degree = mat.sum(axis=-1)
    degree_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    degree_inv_sqrt_mat = np.diag(degree_inv_sqrt)

    agg_mat = degree_inv_sqrt_mat.T @ mat @ degree_inv_sqrt_mat

    return agg_mat