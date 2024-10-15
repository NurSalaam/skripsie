import numpy as np

def applyWindowing(data, windowSize):
    windows = []
    windows = np.lib.stride_tricks.sliding_window_view(data, (windowSize, 1))
    arrWindows = windows.reshape(windows.shape[0], -1)
    return arrWindows


def reverseWindowing(scores, windowSize):
    unwindowed_length = (windowSize - 1) + len(scores)
    mapped = np.full(shape=(unwindowed_length, windowSize), fill_value=np.nan)
    mapped[:len(scores), 0] = scores

    for w in range(1, windowSize):
        mapped[:, w] = np.roll(mapped[:, 0], w)

    if (len(scores) < 200):
        arr = np.nanmean(mapped, axis=1)
    else:
        chunk_size = 150  # Adjust the chunk size as needed

        result_chunks = []
        for i in range(0, len(mapped), chunk_size):
            start_idx = i
            end_idx = i + chunk_size

            chunk_mean = np.nanmean(mapped[start_idx:end_idx], axis=1)
            result_chunks.append(chunk_mean)

        arr = np.concatenate(result_chunks)
    return arr