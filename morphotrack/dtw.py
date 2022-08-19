# from tslearn.metrics import soft_dtw_alignment
import numpy as np
from tqdm import tqdm


def local_duplicate(series, l):
    """
    input:
        series: 1d numpy (dim=n) array
        l: local window to duplicate.
    return:
        (n+l)*(2l+1) 2d array
    """
    append_series = np.insert(series,0,[series[0]]*l)
    append_series = np.insert(append_series,-1,[append_series[-1]]*l)
    local_duplicated_series = np.array([append_series[i:-(l-i)] for i in range(l)])
    local_duplicated_series = local_duplicated_series.T

    return local_duplicated_series


def local_derivative(series, l):
    local_duplicated_series = local_duplicate(series, l)

    # trim the end of the matrix to make the size consistent.
    return np.diff(local_duplicated_series, axis=1)[:-l, :]


def filter_signal(signal, threshold=1e3):
    fourier = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=20e-3/signal.size)
    fourier[frequencies > threshold] = 0

    fft_filtered_signal = np.fft.irfft(fourier)
    # pad to original size
    if fft_filtered_signal.size < signal.size:
        fft_filtered_signal = np.append(fft_filtered_signal,[fft_filtered_signal[-1]]*(signal.size-fft_filtered_signal.size))

    return fft_filtered_signal


def dtw_aligner(standard, target, l, fft_threshold=1e3, gamma=1):
    mtx, score = soft_dtw_alignment(
        local_derivative(filter_signal(standard, fft_threshold), l),
        local_derivative(filter_signal(target, fft_threshold), l),
        gamma=gamma
    )
    idx1 = np.argmax(mtx, axis=0)
    idx2 = np.argmax(mtx, axis=1)

    if idx1.size < target.size:
        idx1 = np.append(idx1, [idx1[-1]]*(target.size-idx1.size))

    if idx2.size < target.size:
        idx2 = np.append(idx2, [idx2[-1]]*(target.size-idx2.size))

    return idx1, idx2, score


def line_wise_dtw_aligner(standard, target, l, fft_threshold=1e3, gamma=1):

    standard2target = np.zeros_like(target)
    target2standard = np.zeros_like(target)
    score_ = np.zeros(target.shape[1])

    for line in tqdm(range(target.shape[1])):
        idx1, idx2, score = dtw_aligner(standard, target[:, line], l, fft_threshold=1e3, gamma=1)

        standard2target[:, line] = idx1
        target2standard[:, line] = idx2
        score_[line] = score

    return standard2target, target2standard, score_

