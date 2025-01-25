import numpy as np
from scipy.signal import butter, filtfilt

class ButterFilterDenoising:
    """
    Class to apply Butterworth filter denoising to a signal.

    Parameters
    ----------
    method : str
        Filtering method ('bandpass', 'lowpass', 'highpass').
    fs : float
        Sample rate of the signal.
    lower_limit : float, optional
        Lower frequency limit for bandpass or highpass filter.
    upper_limit : float, optional
        Upper frequency limit for bandpass or lowpass filter.
    order : int, optional
        Order of the filter. Default is 5.
    """

    def __init__(self, method, fs, lower_limit=None, upper_limit=None, order=5):
        self.method = method
        self.fs = fs
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.order = order
        self.b, self.a = self._design_filter()

    def _design_filter(self):
        """
        Design the Butterworth filter based on the specified method and limits.

        Returns
        -------
        tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Raises
        ------
        ValueError
            If the method is invalid or required limits are not specified.
        """
        nyquist = 0.5 * self.fs

        if self.method == 'bandpass':
            return self._design_bandpass_filter(nyquist)
        elif self.method == 'lowpass':
            return self._design_lowpass_filter(nyquist)
        elif self.method == 'highpass':
            return self._design_highpass_filter(nyquist)
        else:
            raise ValueError("Invalid method. Use 'bandpass', 'lowpass', or 'highpass'.")

    def _design_bandpass_filter(self, nyquist):
        """
        Design a bandpass Butterworth filter.

        Parameters
        ----------
        nyquist : float
            Nyquist frequency.

        Returns
        -------
        tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Raises
        ------
        ValueError
            If lower_limit or upper_limit is not specified.
        """
        if self.lower_limit is None or self.upper_limit is None:
            raise ValueError("Both lower_limit and upper_limit must be specified for bandpass filter")
        low = self.lower_limit / nyquist
        high = self.upper_limit / nyquist
        return butter(self.order, [low, high], btype='band')

    def _design_lowpass_filter(self, nyquist):
        """
        Design a lowpass Butterworth filter.

        Parameters
        ----------
        nyquist : float
            Nyquist frequency.

        Returns
        -------
        tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Raises
        ------
        ValueError
            If upper_limit is not specified.
        """
        if self.upper_limit is None:
            raise ValueError("upper_limit must be specified for lowpass filter")
        normal_cutoff = self.upper_limit / nyquist
        return butter(self.order, normal_cutoff, btype='low')

    def _design_highpass_filter(self, nyquist):
        """
        Design a highpass Butterworth filter.

        Parameters
        ----------
        nyquist : float
            Nyquist frequency.

        Returns
        -------
        tuple
            Numerator (b) and denominator (a) polynomials of the IIR filter.

        Raises
        ------
        ValueError
            If lower_limit is not specified.
        """
        if self.lower_limit is None:
            raise ValueError("lower_limit must be specified for highpass filter")
        normal_cutoff = self.lower_limit / nyquist
        return butter(self.order, normal_cutoff, btype='high')

    def fit(self, signal):
        """
        Filter the input signal using the specified method.

        Parameters
        ----------
        signal : np.ndarray
            Input signal to be filtered.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        return filtfilt(self.b, self.a, signal)
