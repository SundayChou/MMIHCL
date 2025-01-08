"""
Functions for calculating metrics.
"""
import numpy as np
import scanpy as sc
import pandas as pd

from Utils import *
from anndata import AnnData
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import connected_components


def evalAccuracy(matching, labels_1, labels_2):
    """
    Evaluate the cluster level matching accuracy.
    The higher accuracy, the better performance.

    Parameters:
    matching (list): A list of length three.
        The i-th matched pair is (matching[0][i], matching[1][i]).
        The score of the i-th matching pair is matching[2][i].
    labels_1 (list): The list of cell labels for first modality.
    labels_2 (list): The list of cell labels for second modality.

    Returns:
    ACC (float): The cluster level matching accuracy.
    """
    ACC = np.mean([labels_1[i] == labels_2[j] for i, j in zip(matching[0], matching[1])])

    return ACC


def evalFOSCTTM(embeds_1, embeds_2):
    """
    Evaluate the Fraction Of Samples Closer Than True Matching (FOSCTTM).
    The lower FOSCTTM, the better performance.

    Parameters:
    embeds_1 (numpy.ndarray): The cell embeddings of first modality.
    embeds_2 (numpy.ndarray): The cell embeddings of second modality.

    Returns:
    FOSCTTM (float): The fraction to evaluate single-cell-level alignment accuracy.
    """
    dist = correlationDistance(embeds_1, embeds_2)
    ar = np.arange(dist.shape[0])

    mask = (dist.T < dist[ar, ar])
    FOSCTTM = np.mean(np.mean(mask, axis=0))

    return FOSCTTM


def evalGraphConnectivity(mat, anno):
    """
    Evaluate the graph connectivity (GC) of embedding matrix.
    The higher GC, the better performance.

    Parameters:
    mat (numpy.ndarray): The cell embedding matrix.
    anno (numpy.ndarray): The cell annotation array.

    Returns:
    GC (float): The graph connectivity of embedding matrix.
    """
    mat = AnnData(X=mat, dtype=mat.dtype)
    sc.pp.neighbors(mat, n_pcs=0, use_rep="X")
    
    connects = []
    for i in np.unique(anno):
        submat = mat[anno == i]
        _, c = connected_components(submat.obsp['connectivities'], connection='strong')
        counts = pd.value_counts(c)
        connects.append(counts.max() / counts.sum())
    GC = np.mean(connects).item()
    
    return GC


def evalAvgSilhouetteWidth(mat, anno):
    """
    Evaluate the average silhouette width (ASW) of embedding matrix.
    The higher ASW, the better performance.

    Parameters:
    mat (numpy.ndarray): The cell embedding matrix.
    anno (numpy.ndarray): The cell annotation array.

    Returns:
    ASW (float): The ASW of embedding matrix.
    """
    ASW = (silhouette_score(mat, anno).item() + 1) / 2

    return ASW