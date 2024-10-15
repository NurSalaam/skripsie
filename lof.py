# from cuml.neighbors import NearestNeighbors as NearestNeighborsGPU
from sklearn.neighbors import NearestNeighbors as NearestNeighborsCPU
from sklearn.preprocessing import minmax_scale
import numpy as np
import time
from utilities import applyWindowing, reverseWindowing
from dataloader import TimeSeries

class LocalOutlierFactor:
    """
    A class used to house the local outlier factor algorithm.

    ...

    Attributes
    ----------
    windowSize : int
        an integer which determines the subsequence length
    neighbors : int
        the number of neighbors used in kNN
    algorithm : string
        algorithm used to compute the nearest neighbors
    leafSize : int
        leaf size used for BallTree or KDTree in kNN
    distanceMetric : int
        metric to use for distance computation
    gpu : boolean
        flag for whether the GPU version of kNN is used

    Methods
    -------
    fit(tsObject=None)
        Empty method for consistency.
    predict(tsObject=None)
        Executes the local outlier factor algorithm on the tsObject.
    toString():
        Returns a name describing the algorithm configuration.
    """
    def __init__(self, windowSize = 100, neighbors = 20, algorithm = "auto", leafSize = 30, distanceMetric = "minkowski", gpu=False):
        """
        Constructs all the necessary attributes for the LocalOutlierFactor object.

        Parameters
        ----------
            windowSize : int
                an integer which determines the subsequence length
            neighbors : int
                the number of neighbors used in kNN
            algorithm : string
                algorithm used to compute the nearest neighbors
            leafSize : int
                leaf size used for BallTree or KDTree in kNN
            distanceMetric : int
                metric to use for distance computation
            gpu : boolean
                flag for whether the GPU version of kNN is used
        Returns
        -------
            None

        """

        self.windowSize = windowSize
        self.neighbors = neighbors
        self.algorithm = algorithm
        self.leafSize = leafSize
        self.distanceMetric = distanceMetric
        self.gpu = gpu
    
    def fit(self, tsObject: TimeSeries):
        """
        Empty method to keep consistency with other scoring algorithms.

        Parameters
        ----------
            tsObject : TimeSeries
                the TimeSeries object used to store the data and anomalies
        Returns
        -------
            None
        """

        pass
    
    def predict(self, tsObject: TimeSeries):
        """
        Executes the local outlier factor algorithm on the TimeSeries object.

        Parameters
        ----------
            tsObject : TimeSeries
                the TimeSeries object used to store the data and anomalies
        
        Returns
        -------
            tsObject : TimeSeries
                the TimeSeries object used to store the data and anomalies
            totalTime : float
                the total time elapsed during the execution of the algorithm.
        """
        
        # Select the RAPIDS GPU implementation or sklearn CPU implementation
        if (self.gpu == True):
            knn = NearestNeighborsGPU(n_neighbors=self.neighbors+1, algorithm=self.algorithm, metric=self.distanceMetric)
        else:
            knn = NearestNeighborsCPU(n_neighbors=self.neighbors+1, algorithm=self.algorithm, leaf_size=self.leafSize, metric=self.distanceMetric)
        start = time.time()

        #TODO: Figure out if scaling the data is actually necessary to avoid underflow
        
        if (type(tsObject) != np.ndarray):
            testData = tsObject.testData.to_numpy()
        else:
            testData = tsObject

        arrData = applyWindowing(testData, self.windowSize) * 10000

        knn.fit(arrData)
        dist, indices = knn.kneighbors(arrData)
        dist = dist[:, 1:]
        indices = indices[:, 1:]

        # Calculate lrd for all points
        dist_k = dist[indices, self.neighbors - 1]
        reach_dist_array = np.maximum(dist, dist_k)

        # 1e-10 to avoid `nan' when nb of duplicates > n_neighbors_:
        lrd =  1.0 / (np.mean(reach_dist_array, axis=1) + 1e-10)

        # Calculate the local outlier factor by comparing the point's lrd with the neighbourhood average
        lofScores = np.mean(lrd[indices] / lrd[:, np.newaxis], axis=1)

        scores = reverseWindowing(lofScores, self.windowSize)
        
        end = time.time()

        flattenedScores = minmax_scale(scores.reshape(-1, 1)).flatten()

        if (type(tsObject) != np.ndarray):
            tsObject.testData["Score"] = flattenedScores

        totalTime = end - start

        if (type(tsObject) == np.ndarray):
            return flattenedScores
        
        return tsObject, totalTime

    def toString(self):
        """
        Returns the name of the local outlier factor configuration.
        
        Returns
        -------
            name : str
                the name of the local outlier factor configuration.
        """

        name = "lof_w" + str(self.windowSize) + "_n" + str(self.neighbors)
        return name