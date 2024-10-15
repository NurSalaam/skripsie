import pandas as pd
"""


"""
class DataLoader:
    def __init__(self):
        """
        Initialize the TimeSeriesDataLoader.
        """
    
    def load_file(self, file_path, tolerance=100):
        """
        Load a single time series from a CSV file.

        Args:
            file_path (str): The path to the CSV file containing time series data.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the loaded time series data.
        """

        try:
            # Open the file in read mode ('r')
            with open(file_path, 'r') as file:
                # Read the entire file into a string
                file_contents = file.readlines()
            
            counter = 0
            for line in file_contents:
                line = line.rstrip('\n')
                counter += 1

                if (line == "###"):
                    skip = counter
                    break
                
                lineData = line.split(":")
                property = lineData[0]
                content = lineData[1]

                if property == "name":
                    name = content
                elif property == "trainingsteps":
                    trainLength = int(content)
                elif property == "anomalies":
                    anomalies = []
                    splitContent = content.split(",")

                    anomalyID = 0
                    for text in splitContent:
                        txtArr = text.split('_')
                        anomaly = Anomaly(anomalyID, int(txtArr[0]), int(txtArr[1]) - tolerance - trainLength, int(txtArr[2]) + tolerance - trainLength)
                        anomalies.append(anomaly)
                        anomalyID += 1
                else:
                    print("Error.")

            data = pd.read_csv(file_path, skiprows=skip)
            totalLength = len(data)
            testLength = totalLength-trainLength
            trainData = data[:trainLength]
            testData = data[trainLength:]
            tsObject = TimeSeries(name, trainData, testData, anomalies, trainLength, testLength)
            assert (len(trainData) == trainLength)
            assert (len(testData) == testLength)
            return tsObject

        except FileNotFoundError:
            print(f"File not found: {file_path}")

class TimeSeries:
    def __init__(self, name, trainData, testData, anomalies, trainLength, testLength):
        """
        A class used to represent a single time series anomaly detection problem.
        """

        self.name = name
        self.trainData = trainData
        self.testData = testData
        self.anomalies = anomalies
        self.trainLength = trainLength
        self.testLength = testLength

    def toString(self):
        return self.name
    
class Anomaly:
    def __init__(self, id, channel, start, end):
        """
        A class used to represent a single anomaly.
        """

        self.id = id
        self.channel = channel
        self.start = start
        self.end = end

    def toString(self):
        print("Anomaly " + str(self.id) + ": Channel " + str(self.channel) + ", " + str(self.start) + "-" + str(self.end))