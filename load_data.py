"""
Processes raw gesture data from CSV files.

Functions:
    norm(arr): Normalizes the input array.

Variables:
    segment_length (int): The length of each data segment.
    window_size (int): The size of the window around each detected peak.
    samples (int): The number of samples in each segment.
    statistics (numpy.ndarray): Array to store the processed data.
    total (int): Total number of gestures.
    
Author: Yingxin Gao
Last modified: 12/24/2024
Version: 1.0

Dependencies:
    - numpy
    - pandas
    - scipy
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

segment_length = 3200
window_size = 18
samples = segment_length // window_size
statistics = np.zeros((5, 5, samples, window_size * 2))
gestures = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
total = 5


def norm(arr):
    """Normalize the input array."""
    if np.std(arr) == 0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)


# 读取data目录下的csv文件
for i, gesture in enumerate(gestures):
    df = pd.read_csv(f'data/{gestures[i]}.csv')

    # 将数据转换为numpy数组
    data = df.values
    data[data < 1e-4] = 0

    # 根据不同手指选择不同的峰值检测列
    ch_idx = 5 if i in (0, 1, 4) else 1  # Thumb, Index, Little -> Ch6; Middle, Ring -> Ch2
    peaks, _ = find_peaks(data[:segment_length, ch_idx])

    for k, peak in enumerate(peaks):
        start = max(0, peak - window_size)
        end = min(len(data), peak + window_size)

        if end - start == window_size * 2 and k < samples:
            # Assign data to the statistics array for all 5 channels
            for ch in range(5):
                statistics[i, ch, k, :] = data[start:end, ch]

    # Normalize each channel for the current gesture
    for ch in range(5):
        statistics[i, ch, :, :] = norm(statistics[i, ch, :, :])

np.save('data.npy', statistics)
