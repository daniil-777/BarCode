import hnswlib
import numpy as np

def HNSWNearestNeighbors(X, k=2):
    '''For a given set of points X finds k nearest neighbors for every point
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Point data to find nearest neighbors,
        where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    labels : array-like, shape (n_samples, k)
        indices of k nearest neighbors for every point
    distances : array-like, shape (n_samples, k)
        distances to k nearest neighbors in increasing order
    '''
    X_float32 = np.array(X, dtype=np.float32)
    X_labels = np.arange(len(X))
    p = hnswlib.Index(space='l2', dim=X.shape[1]) 
    p.init_index(max_elements=len(X), ef_construction=200, M=16)
    p.add_items(X_float32, X_labels)
    p.set_ef(120 + k)
    indexes, distance = p.knn_query(X, k=k + 1)
    return indexes[:,1:], distance[:,1:]**0.5
