from sklearn.cluster import KMeans as SkKMeans
from sklearn.preprocessing import minmax_scale
import numpy as np
import time
from utilities import applyWindowing, reverseWindowing
from dataloader import TimeSeries

class KMeans:
    """
    A class used to implement k-means clustering for anomaly detection.

    Attributes
    ----------
    windowSize : int
        The size of the sliding window used for feature extraction.
    n_clusters : int
        The number of clusters to form in the k-means algorithm.


    Methods
    -------
    fit(tsObject: TimeSeries)
        Fits the k-means model to the data.
    predict(tsObject: TimeSeries)
        Predicts anomaly scores for the time series data.
    toString()
        Returns a string representation of the KMeans configuration.
    """

    def __init__(self, windowSize=100, n_clusters=8):
        """
        Initializes the KMeans object with specified parameters.

        Parameters
        ----------
        windowSize : int, optional
            The size of the sliding window (default is 100).
        n_clusters : int, optional
            The number of clusters to form (default is 8).
        """
        self.windowSize = windowSize
        self.n_clusters = n_clusters
        self.model = SkKMeans(n_clusters=self.n_clusters, n_init="auto")
    
    def fit(self, tsObject: TimeSeries):
        """
        Fits the k-means model to the windowed time series data.

        Parameters
        ----------
        tsObject : TimeSeries
            The TimeSeries object containing the data to fit.

        Returns
        -------
        None
        """
    
    def predict(self, tsObject: TimeSeries):
        """
        Predicts anomaly scores for the time series data.

        Parameters
        ----------
        tsObject : TimeSeries
            The TimeSeries object containing the data to predict on.

        Returns
        -------
        tsObject : TimeSeries
            The input TimeSeries object with added anomaly scores.
        totalTime : float
            The total execution time of the prediction.
        """
        start = time.time()
        arrData = applyWindowing(tsObject.testData, self.windowSize)
        self.model.fit(arrData)
        self.model.predict(arrData)
        
        clusterCenters = self.model.cluster_centers_[self.model.labels_]
        distances = np.linalg.norm(clusterCenters - arrData, axis=1)
        scores = reverseWindowing(distances, self.windowSize)
        end = time.time()
        totalTime = end - start

        flattenedScores = minmax_scale(scores.reshape(-1, 1)).flatten()
        tsObject.testData["Score"] = flattenedScores
        return tsObject, totalTime

    def toString(self):
        """
        Returns a string representation of the KMeans configuration.

        Returns
        -------
        name : str
            The name of the KMeans configuration.
        """
        name = f"km_w{self.windowSize}_k{self.n_clusters}"
        return name
    


