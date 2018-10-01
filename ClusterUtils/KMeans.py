import copy
import pandas as pd
import numpy as np
import random
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_kmeans_

# https://en.wikipedia.org/wiki/K-means_clustering#Algorithms
# Pretty much using for pseudocode sections
debug = False

def distance(point, centroid):
    # Euclidean distance. No root because it will always be the same.
    dist = 0
    for val in range(len(point)):
        dist += (point[val] - centroid[val]) ** 2
    return dist

def get_closest_centroid(point, centroids):
    #return an int from 0 to len(centroids)
    minimum = [-1,None]
    for cent in range(len(centroids)):
        #calculate the distance in easiest way
        dist = distance(point, centroids[cent])
        if minimum[0] == -1 or dist < minimum[0]:
            minimum[0] = dist
            minimum[1] = cent
    return minimum[1]

def get_min_and_max(X):
    # Finds the ranges for all the data. Helpful for generating random centroids
    to_return = [None] * X.shape[1]

    for point in X:
        for minmax in range(X.shape[1]):
            if to_return[minmax] == None:
                to_return[minmax] = [point[minmax], point[minmax]]
            else:
                to_return[minmax][0] = min(to_return[minmax][0], point[minmax])
                to_return[minmax][1] = max(to_return[minmax][0], point[minmax])
    print(to_return)
    return to_return

def fitness(X, centroids, labels):
    to_return = 0 # Where we will be putting the distance
    # s = (b – a) / max(a,b)
    # a is avg distance to in-cluster
    # Extraordinarily inefficient because of being O(N^2)

    for point1 in range(X.shape[0]):
        a = 0
        a_count = 0
        b_list = [0] * len(centroids)
        b_counts = [0] * len(centroids) #gotta count the number of points per outgroup yo
        for point2 in range(X.shape[0]):
            if labels[point1] == labels[point2]:
                a += distance(X[point1],X[point2])
                a_count += 1
            else:
                b_list[labels[point2]] += distance(X[point1],X[point2])
                b_counts[labels[point2]] += 1
        a = a/a_count
        b_list.pop(labels[point1]) # Avoid the 0/0 from ingroup
        b_counts.pop(labels[point1])

        b = min([b_list[x] / (b_counts[x] + 1) for x in range(len(b_list))])

        s = (b - a) / max(a,b)
        to_return += s
    to_return = to_return / X.shape[0]
    return to_return

def k_means(X, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300, verbose=False):

    # Implement.

    # Input: np.darray of samples

    # Return the following:
    #
    # 1. labels: An array or list-type object corresponding to the predicted
    #  cluster numbers,e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # 2. centroids: An array or list-type object corresponding to the vectors
    # of the centroids, e.g., [[0.5, 0.5], [-1, -1], [3, 3]]
    # 3. inertia: A number corresponding to some measure of fitness,
    # generally the best of the results from executing the algorithm n_init times.
    # You will want to return the 'best' labels and centroids by this measure.

    labels = [None] * X.shape[0]
    centroids = [] #List of essentially points
    inertia = -1 # measure of fitness

    for loop in range(n_init): # Do a bunch of times and take 'best' by some measure
        #Initialize the centroids etc for current iteration
        cur_centroids = [] * n_clusters
        ranges = get_min_and_max(X) #So we know where centroids can appear
        i = 0
        while i < n_clusters:
            i += 1
            #Gotta make that uniform distribution yo.
            location = [random.uniform(ranges[item][0], ranges[item][1]) for item in range(X.shape[1])]
            cur_centroids.append(location)
        if debug: print(cur_centroids)
        cur_labels = [None] * X.shape[0]
        cur_inertia = 0 # measure of fitness

        x = 0
        last_centroids = None
        while x < max_iter and cur_centroids != last_centroids:
            last_centroids = copy.deepcopy(cur_centroids) #Gotta be deep
            x += 1
            #Assignment Step
                # Assign each observation to the cluster whose mean has the least squared Euclidean distance,
            counts = [0] * n_clusters # keep track of number in each cluster
            sums = [[0] * X.shape[1] for c in range(n_clusters)] # For an average of the next set of centroids
            for sample in range(X.shape[0]):
                min_centroid = get_closest_centroid(X[sample], cur_centroids)
                cur_labels[sample] = min_centroid
                #Update sums for this centroid
                counts[min_centroid] += 1
                for dimension in range(X.shape[1]):
                    sums[min_centroid][dimension] += X[sample][dimension]

            #Reset centroid step

            for centroid in range(len(cur_centroids)):
                if counts[centroid] == 0:
                    #Reassign centroid to new random point
                    cur_centroids[centroid] = []
                    location = [random.uniform(ranges[item][0], ranges[item][1]) for item in range(X.shape[1])]
                    cur_centroids[centroid] = location
                else:
                    for dimension in range(X.shape[1]):
                        cur_centroids[centroid][dimension] = sums[centroid][dimension] / counts[centroid]

            cur_inertia = fitness(X, cur_centroids, cur_labels)
            if inertia == -1 or cur_inertia < inertia:
                inertia = cur_inertia
                labels = cur_labels
                centroids = cur_centroids

    return labels, centroids, inertia


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class KMeans(SuperCluster):
    """
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'random', 'k-means++', 'global'}
        Method for initialization, defaults to 'random'.
    algorithm : {'lloyds', 'hartigans'}
        Method for determing algorithm, defaults to 'lloyds'.
    n_init : int, default: 1
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
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

    def __init__(self, n_clusters=3, init='random', algorithm='lloyds', n_init=1, max_iter=300,
                 csv_path=None, keep_dataframe=True, keep_X=True, verbose=False):
        self.n_clusters = n_clusters
        self.init = init
        self.algorithm = algorithm
        self.n_init = n_init
        self.max_iter = max_iter
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X
        self.verbose = verbose

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels, self.centroids, self.inertia = \
            k_means(X, n_clusters=self.n_clusters, init=self.init, algorithm=self.algorithm,
                    n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose)
        print(self.init + " k-means finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels, centroids=self.centroids)
        else:
            print('No data to plot.')

    def save_plot(self, name = 'kmeans_plot'):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_kmeans_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_kmeans_(X=self.X, labels=self.labels,
                            centroids=self.centroids, save=True, n=name)
        else:
            print('No data to plot.')
