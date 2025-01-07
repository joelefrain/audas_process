import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def filter_signal(method, signal, fs, lower_limit=None, upper_limit=None, order=5):
    """
    Filter the input signal using the specified method.

    Parameters
    ----------
    method : str
        Filtering method ('bandpass', 'lowpass', 'highpass').
    signal : np.ndarray
        Input signal to be filtered.
    fs : float
        Sample rate of the signal.
    lower_limit : float, optional
        Lower frequency limit for bandpass or highpass filter.
    upper_limit : float, optional
        Upper frequency limit for bandpass or lowpass filter.
    order : int, optional
        Order of the filter. Default is 5.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    if method == 'bandpass':
        if lower_limit is None or upper_limit is None:
            raise ValueError("Both lower_limit and upper_limit must be specified for bandpass filter")
        b, a = butter_bandpass(lower_limit, upper_limit, fs, order=order)
    elif method == 'lowpass':
        if upper_limit is None:
            raise ValueError("upper_limit must be specified for lowpass filter")
        b, a = butter_lowpass(upper_limit, fs, order=order)
    elif method == 'highpass':
        if lower_limit is None:
            raise ValueError("lower_limit must be specified for highpass filter")
        b, a = butter_highpass(lower_limit, fs, order=order)
    else:
        raise ValueError("Invalid method. Use 'bandpass', 'lowpass', or 'highpass'.")

    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
