from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest as skIsoForest
import time

from dataloader import TimeSeries
from utilities import applyWindowing, reverseWindowing

class IsolationForest:
    """
    A class used to implement Isolation Forest for anomaly detection.

    Attributes
    ----------
    windowSize : int
        The size of the sliding window used for feature extraction.
    n_estimators : int
        The number of base estimators in the ensemble.

    Methods
    -------
    fit(tsObject: TimeSeries, trainData: bool = False)
        Fits the Isolation Forest model to the data.
    predict(tsObject: TimeSeries)
        Predicts anomaly scores for the time series data.
    toString()
        Returns a string representation of the IsolationForest configuration.
    """

    def __init__(self, windowSize=75, n_estimators=100):
        """
        Initializes the IsolationForest object with specified parameters.

        Parameters
        ----------
        windowSize : int, optional
            The size of the sliding window (default is 75).
        n_estimators : int, optional
            The number of base estimators in the ensemble (default is 100).
        """
        self.windowSize = windowSize
        self.n_estimators = n_estimators

        # Don't fully understand the error, but it seems that parrallelising the code causes the self.__model to be None if I instantiate it here.  So instead, I move it to fit
        self.__model = None
    
    def fit(self, tsObject: TimeSeries):
        """
        Fits the Isolation Forest model to the windowed time series data.

        Parameters
        ----------
        tsObject : TimeSeries
            The TimeSeries object used to store the data and anomalies.
        trainData : bool, optional
            If True, fits on training data; otherwise, fits on test data (default is False).

        Returns
        -------
        fitTime : float
            The time taken to fit the model.
        """        

        start = time.time()

    
        self.__model = skIsoForest(n_estimators=self.n_estimators)
        arrData = applyWindowing(tsObject.testData, self.windowSize)
        self.__model.fit(arrData)
        end = time.time()

        fitTime = end - start
        return fitTime
    
    def predict(self, tsObject):
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
        predictTime : float
            The time taken to make predictions.
        """
        start = time.time()
        arrData = applyWindowing(tsObject.testData, self.windowSize)
        isoForestScores = self.__model.decision_function(arrData)
        scores = reverseWindowing(isoForestScores, self.windowSize)
        end = time.time()

        # multiply by -1, since the lower the score the more anomalous, however we want
        # to comply to the convention that higher values are more anomalous
        tsObject.testData["Score"] = -1*scores

        #! Think about adding an `assert` statement to check that the length of 'Scores' returned by the esitmator is the same as the original `testData` array
        tsObject.testData["Score"] = MinMaxScaler().fit_transform(tsObject.testData["Score"].values.reshape(-1,1))
        predictTime = end - start
        return tsObject, predictTime
    
    def toString(self):
        """
        Returns a string representation of the IsolationForest configuration.

        Returns
        -------
        str
            The name of the IsolationForest configuration.
        """
        return f"if_w{self.windowSize}_n{self.n_estimators}"