import pandas as pd
import time
from ClusterUtils.ClusterPlotter import _plot_cvnn_
from ClusterUtils.ClusterPlotter import _plot_silhouette_

# http://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf

def distance(point, centroid):
    # Euclidean distance. No root because it will always be the same.
    dist = 0
    for val in range(len(point)):
        dist += (point[val] - centroid[val]) ** 2
    return dist

def silhouette(X, centroids, labels):
    # This one creates a column for the pandas dataframe
    # So not averages.


    to_return = [] # Where we will be putting the distance
    # s = (b – a) / max(a,b)
    # a is avg distance to in-cluster
    # Extraordinarily inefficient because of being O(N^2)
    # Considering only working off a sample of the data to cut down on time.

    for point1 in range(X.shape[0]):
        a = 0
        a_count = 0
        b_list = [0] * centroids
        b_counts = [0] * centroids #gotta count the number of points per outgroup yo
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
        to_return.append(s)
    #to_return = to_return / X.shape[0]
    return to_return

def tabulate_silhouette(datasets, cluster_nums):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.

    #s = (b - a) / max(a, b)
    # a is average distance to ingroup
    # b is average to closest outgroup. (so second-best)
    # 0-1, want close to 1. Source, class slides
    # Average it then.

    # I wrote the thing in the KMeans file first,
    sil = None
    for d in range(len(datasets)):
        sil = silhouette(datasets[d][:-1], cluster_nums[d], datasets[d][-1])


    cols = ['Clusters', 'SILHOUETTE_IDX']
    df = pd.DataFrame(data=sil,
                  index=range(datasets[0].shape[0]),
                  columns=cols)
    return df



    #plt.plot(silhouette_table['CLUSTERS'], silhouette_table['SILHOUETTE_IDX'], label='Silhouette Index')

     #   sil = (b - a) / max(a, b)


def tabulate_cvnn(datasets, cluster_nums, k_vals):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]

    # Return a pandas DataFrame corresponding to the results.

    return None


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class InternalValidator:
    """
    Parameters
    ----------
    datasets : list or array-type object, mandatory
        A list of datasets. The final column should cointain predicted labeled results
        (By default, the datasets generated are pandas DataFrames, and the final
        column is named 'CLUSTER')
    cluster_nums : list or array-type object, mandatory
        A list of integers corresponding to the number of clusters used (or found).
        Should be the same length as datasets.
    k_vals: list or array-type object, optional
        A list of integers corresponding to the desired values of k for CVNN
        """

    def __init__(self, datasets, cluster_nums, k_vals=[1, 5, 10, 20]):
        self.datasets = list(map(lambda df : df.drop('CENTROID', axis=0), datasets))
        self.cluster_nums = cluster_nums
        self.k_vals = k_vals

    def make_cvnn_table(self):
        start_time = time.time()
        self.cvnn_table = tabulate_cvnn(self.datasets, self.cluster_nums, self.k_vals)
        print("CVNN finished in  %s seconds" % (time.time() - start_time))

    def show_cvnn_plot(self):
        _plot_cvnn_(self.cvnn_table)

    def save_cvnn_plot(self, name='cvnn_plot'):
        _plot_cvnn_(self.cvnn_table, save=True, n=name)

    def make_silhouette_table(self):
        start_time = time.time()
        self.silhouette_table = tabulate_silhouette(self.datasets, self.cluster_nums)
        print("Silhouette Index finished in  %s seconds" % (time.time() - start_time))

    def show_silhouette_plot(self):
        _plot_silhouette_(self.silhouette_table)

    def save_silhouette_plot(self, name='silhouette_plot'):
        _plot_silhouette_(self.cvnn_table, save=True, n=name)

    def save_csv(self, cvnn=False, silhouette=False, name='internal_validator'):
        if cvnn is False and silhouette is False:
            print('Please pass either cvnn=True or silhouette=True or both')
        if cvnn is not False:
            filename = name + '_cvnn_' + (str(round(time.time()))) + '.csv'
            self.cvnn_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if silhouette is not False:
            filename = name + '_silhouette_' + (str(round(time.time()))) + '.csv'
            self.silhouette_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if cvnn is False and silhouette is False:
            print('No data to save.')
