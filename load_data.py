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
samples = int(segment_length / window_size)
statistics = np.zeros((5, 5, samples, window_size*2))
total=5

def norm(arr):
    """
    Normalize the input array.

    Args:
        arr (numpy.ndarray): Input array to be normalized.

    Returns:
        numpy.ndarray: Normalized array.
    """
    if np.std(arr)==0:
        return arr
    return (arr - np.mean(arr)) / np.std(arr)

# 修改文件读取路径，使用data目录
for i in range(5):
    gestures = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    df = pd.read_csv(f'data/{gestures[i]}.csv') # 使用正确的文件路径

    # 将数据转换为numpy数组
    data = df.values
    data[data < 1e-4] = 0

    # 根据不同手指选择不同的峰值检测列
    if i in (0, 1, 4):  # Thumb, Index, Little
        peaks, _ = find_peaks(data[:segment_length, 5])  # 使用第6列(Ch6)检测峰值
    else:  # Middle, Ring
        peaks, _ = find_peaks(data[:segment_length, 1])  # 使用第2列(Ch2)检测峰值

    for k, peak in enumerate(peaks):
        start = max(0, peak - window_size)
        end = min(len(data), peak + window_size)

        if end >= data.shape[0] or k >= statistics.shape[2] or end - start != window_size * 2:
            continue

        # 从6个通道中选择需要的5个通道数据
        statistics[i, 0, k, :] = data[start:end, 0]  # Ch1
        statistics[i, 1, k, :] = data[start:end, 1]  # Ch2
        statistics[i, 2, k, :] = data[start:end, 2]  # Ch3
        statistics[i, 3, k, :] = data[start:end, 3]  # Ch4
        statistics[i, 4, k, :] = data[start:end, 4]  # Ch5

    statistics[i, 0, :, :] = norm(statistics[i, 0, :, :])
    statistics[i, 1, :, :] = norm(statistics[i, 1, :, :])
    statistics[i, 2, :, :] = norm(statistics[i, 2, :, :])
    statistics[i, 3, :, :] = norm(statistics[i, 3, :, :])
    statistics[i, 4, :, :] = norm(statistics[i, 4, :, :])

np.save('data.npy', statistics)
