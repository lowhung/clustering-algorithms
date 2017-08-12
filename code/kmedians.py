import numpy as np
import utils

class Kmedians:

    def __init__(self, k):
        self.k = k
        self.medians = []

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        medians = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            medians[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = utils.euclidean_dist_squared(X, medians)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                medians[kk] = np.median(X[y==kk], axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.medians = medians

    def predict(self, X):
        medians = self.medians
        dist2 = utils.euclidean_dist_squared(X, medians)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        N, D = X.shape
        medians = self.medians
        closest_median_indexes = self.predict(X)

        error = 0
        for i in range(medians.shape[0]):
            error += np.sum(utils.euclidean_dist_squared(X[closest_median_indexes==i], medians[[i]]))
        return error
