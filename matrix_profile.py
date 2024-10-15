# import numpy as np
# from sklearn.preprocessing import minmax_scale
# import time
# import pyscamp
# from dataloader import TimeSeries

# class MatrixProfile:
#     def __init__(self, windowSize=100):
#         self.windowSize = windowSize

#     def fit(self, tsObject: TimeSeries):
#         # Empty method for consistency
#         pass

#     def predict(self, tsObject: TimeSeries):
#         start = time.time()

#         if isinstance(tsObject, np.ndarray):
#             data = tsObject
#             print("is ndarray")
#         else:
#             data = tsObject.testData.values
            
#         profile, index = pyscamp.selfjoin(data.flatten(), self.windowSize)
#         scores = profile

#         # Check for NaN values in the profile
#         if np.isnan(scores).any():
#             # print("Warning: NaN values detected in the matrix profile")
#             # You might want to handle this case, e.g., by replacing NaNs with a default value
#             scores = np.nan_to_num(scores, nan=np.nanmin(scores)) # set it to the min valuse, since nan will only occur in normal regions

#         scores_range = np.max(scores) - np.min(scores)

#         # Normalize scores to a 0-1 range, with 1 being the most anomalous
#         if scores_range > 1e-10:  # Check if the range is not too small
#             normalized_scores = minmax_scale(scores.reshape(-1, 1)).flatten()
#         else:
#             print("Warning: Score range is too small, using uniform scores")
#             normalized_scores = np.ones_like(scores)  # or np.zeros_like(scores), depending on your preference

#         totalTime = time.time() - start

#         padding_length = data.shape[0] - len(normalized_scores)
#         last_value = normalized_scores[-1]
#         padded_scores = np.pad(normalized_scores, (0, padding_length), 'constant', constant_values=last_value)

#         # Assign padded scores to testData
#         if not isinstance(tsObject, np.ndarray):
#             tsObject.testData["Score"] = padded_scores
#             return tsObject, totalTime
#         else:
#             return padded_scores, totalTime

#     def toString(self):
#         return f"mp_w{self.windowSize}"


import numpy as np
from sklearn.preprocessing import minmax_scale
import time
import pyscamp
from dataloader import TimeSeries

class MatrixProfile:
    def __init__(self, windowSize=100):
        self.windowSize = windowSize

    def fit(self, tsObject: TimeSeries):
        # Empty method for consistency
        pass

    def predict(self, tsObject: TimeSeries):
        start = time.time()

        if isinstance(tsObject, np.ndarray):
            data = tsObject
        else:
            data = tsObject.testData.values
            
        profile, index = pyscamp.selfjoin(data.flatten(), self.windowSize)
        scores = profile

        # Check for NaN values in the profile
        if np.isnan(scores).any():
            scores = np.nan_to_num(scores, nan=np.nanmin(scores))

        scores_range = np.max(scores) - np.min(scores)

        # Normalize scores to a 0-1 range, with 1 being the most anomalous
        if scores_range > 1e-10:  # Check if the range is not too small
            normalized_scores = minmax_scale(scores.reshape(-1, 1)).flatten()
        else:
            print("Warning: Score range is too small, using uniform scores")
            normalized_scores = np.ones_like(scores)  # or np.zeros_like(scores), depending on your preference

        # Handle the case where normalized_scores is longer than data
        if len(normalized_scores) > data.shape[0]:
            normalized_scores = normalized_scores[:data.shape[0]]
        elif len(normalized_scores) < data.shape[0]:
            padding_length = data.shape[0] - len(normalized_scores)
            last_value = normalized_scores[-1]
            normalized_scores = np.pad(normalized_scores, (0, padding_length), 'constant', constant_values=last_value)

        totalTime = time.time() - start

        # Assign normalized scores to testData
        if not isinstance(tsObject, np.ndarray):
            tsObject.testData["Score"] = normalized_scores
            return tsObject, totalTime
        else:
            return normalized_scores, totalTime

    def toString(self):
        return f"mp_w{self.windowSize}"