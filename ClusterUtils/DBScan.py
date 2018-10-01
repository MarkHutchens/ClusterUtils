import pandas as pd
import numpy as np
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_

'''
Looked at Wikipedia: https://en.wikipedia.org/wiki/DBSCAN

How DBSCAN works:

Find cores and borders and noise
    cores are ones close enough to a bunch of other points. ie in a dense region
When you find a core point:
    Jump to all its neighbors to see if they are core points too. Repeat

'''

debug = False

def distance(point1, point2, size):
    #so many features, so little time.
    dis = 0
    for feat in range(size):
        dis += (point1[feat] - point2[feat]) **2
    return dis

def range_query(X, point1, core, eps = 1):
    #Name taken from Wikipedia. Easier to understand dbscan with this extracted
    shape = X.shape
    neighbors = [] #To keep track of neighbors
    count = 0 # Easier -1 than handling counting itself
    if debug: print(X[point1], end=': ')
    for point2 in range(shape[0]):
        if point1 != point2:
            if distance(X[point1], X[point2], shape[1]) <= eps:
                count += 1
                neighbors.append(point2)
    if debug: print(count)
    return(neighbors)



def dbscan(X: np.ndarray, eps=1, min_points=10, verbose=False):
    # eps = max distance between same-cluster points
    # min-points is the cutoff for a dense region
    # Implement.

    # Input: np.ndarray of samples, X

    # Return a array or list-type object corresponding to the predicted cluster
    # numbers, e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]
    eps = eps ** 2 # Otherwise we'd need to square root every distance
    shape = X.shape
    core = [None] * X.shape[0]

    cluster = 0 #Will increment with each cluster we find

    for point1 in range(shape[0]):
        #First off, determine if each point is core.
        if core[point1] != None: #If we've already labelled it, don't bother again.
            continue
        neighbors = range_query(X, point1, core, eps) #get a list of neighbors
        count = len(neighbors)

        if count >= min_points: #Noice, got a core point!
            cluster += 1
            core[point1] = cluster
            if debug: print(len(neighbors), end = ' to ')
            for n in neighbors:
                if core[n] == 0:
                    core[n] = cluster #Noise to edge, w00t
                elif core[n] != None:
                    continue
                #Now the wonky bit. We append points to the neighbors list!
                core[n] = cluster
                n_neighbors = range_query(X, n, core, eps)
                #print(neighbors)
                if len(n_neighbors) >= min_points:
                    neighbors.extend(n_neighbors)
            if debug: print(len(neighbors))
        else: # Noise
            core[point1] = 0
    if debug: print(len(core), core)
    return core


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class DBScan(SuperCluster):
    """
    Perform DBSCAN clustering from vector array.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    csv_path : str, default: None
        Path to file for dataset csv
    keep_dataframe : bool, default: True
        Hold on the results pandas DataFrame generated after each run.
        Also determines whether to use pandas DataFrame as primary internal data state
    keep_X : bool, default: True
        Hold on the results generated after each run in a more generic array-type format
        Use these values if keep_dataframe is False
    verbose: bool, default: False
        Optional log level
    """

    def __init__(self, eps=1, min_points=10, csv_path=None, keep_dataframe=True,
                                                    keep_X=True,verbose=False):
        self.eps = eps
        self.min_points = min_points
        self.verbose = verbose
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels = dbscan(X, eps=self.eps, min_points = self.min_points,verbose = self.verbose)
        print("DBSCAN finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels)
        else:
            print('No data to plot.')

    def save_plot(self, name):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels, save=True, n=name)
        else:
            print('No data to plot.')
