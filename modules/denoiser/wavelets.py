import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec

# Auxiliary functions


def energy(x):
    """
    Compute the energy of a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).

    Returns
    -------
    float
        Energy of the input signal.
    """
    return np.dot(x, x)


def euclidean_norm(x):
    """
    Compute the Euclidean norm (p-norm with p=2) of the input signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).

    Returns
    -------
    float
        Euclidean norm of the input signal.
    """
    return np.linalg.norm(x)


def mad(x):
    """
    Estimate the Median Absolute Deviation (MAD).

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).

    Returns
    -------
    float
        Median absolute deviation of the input signal.
    """
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def meanad(x):
    """
    Estimate the Mean Absolute Deviation (MeanAD).

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).

    Returns
    -------
    float
        Mean absolute deviation of the input signal.
    """
    return 1.482579 * np.mean(np.abs(x - np.mean(x)))


def grad_g_fun(x, thr=1):
    """
    Gradient function for thresholding.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).
    thr : float, optional
        Threshold value. Default is 1.

    Returns
    -------
    np.ndarray
        Gradient function result.
    """
    return (x >= thr) * 1 + (x <= -thr) * 1 + (np.abs(x) <= thr) * 0


def nearest_even_integer(n):
    """
    Return the nearest even integer to the input number.

    Parameters
    ----------
    n : int
        Input number.

    Returns
    -------
    int
        Nearest even integer to the input number.
    """
    return n if n % 2 == 0 else n - 1


def dyadic_length(x):
    """
    Return the length and the dyadic length of the input signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).

    Returns
    -------
    tuple
        Length of the input signal and the least power of 2 greater than the length.
    """
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.0)).astype("i")
    return m, j


def soft_hard_thresholding(x, thr=1, method="s"):
    """
    Perform either soft or hard thresholding on the input signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).
    thr : float, optional
        Threshold value. Default is 1.
    method : str, optional
        Thresholding method ('s' for soft, 'h' for hard). Default is 's'.

    Returns
    -------
    np.ndarray
        Thresholded signal.
    """
    if method.lower() == "h":
        return x * (np.abs(x) > thr)
    elif method.lower() == "s":
        return (x >= thr) * (x - thr) + (x <= -thr) * (x + thr) + (np.abs(x) <= thr) * 0
    else:
        raise ValueError(
            "Thresholding method not found! Choose 's' (soft) or 'h' (hard)"
        )


# Main wavelets denoising class


class WaveletDenoising:
    """
    Wavelet denoising class.
    """

    def __init__(
        self,
        normalize=False,
        wavelet="haar",
        level=1,
        thr_mode="soft",
        recon_mode="smooth",
        selected_level=0,
        method="universal",
        energy_perc=0.9,
    ):
        """
        Initialize the WaveletDenoising class.

        Parameters
        ----------
        normalize : bool, optional
            Enable normalization of the input signal into [0, 1]. Default is False.
        wavelet : str, optional
            Type of wavelet to use. Default is 'haar'.
        level : int, optional
            Decomposition level. Default is 1.
        thr_mode : str, optional
            Type of thresholding ('soft' or 'hard'). Default is 'soft'.
        recon_mode : str, optional
            Reconstruction signal extension mode. Default is 'smooth'.
        selected_level : int, optional
            Selected level for thresholding. Default is 0.
        method : str, optional
            Method for threshold determination. Default is 'universal'.
        energy_perc : float, optional
            Energy level retained in the coefficients when using the energy thresholding method. Default is 0.9.
        """
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.thr_mode = thr_mode
        self.selected_level = selected_level
        self.recon_mode = recon_mode
        self.energy_perc = energy_perc
        self.normalize = normalize

        self.filter_ = Wavelet(self.wavelet)  # Wavelet function

        # Check if level is None and set it to 1
        self.nlevel = 1 if level is None else level
        self.normalized_data = None

    def fit(self, signal):
        """
        Execute the denoising algorithm.

        Parameters
        ----------
        signal : np.ndarray
            Noisy input signal.

        Returns
        -------
        np.ndarray
            Denoised signal.
        """
        tmp_signal = self.preprocess(signal)
        coeffs = self.wav_transform(tmp_signal)
        return self.denoise(tmp_signal, coeffs)

    def preprocess(self, signal):
        """
        Remove trends and normalize the input signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).

        Returns
        -------
        np.ndarray
            Detrended (and normalized) signal.
        """
        xhat = detrend(signal)
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            xhat = self.scaler.fit_transform(xhat.reshape(-1, 1))[:, 0]
            self.normalized_data = xhat.copy()
        return xhat

    def std(self, signal, level=None):
        """
        Estimate the standard deviation of the input signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).
        level : int, optional
            Decomposition level. Default is None.

        Returns
        -------
        np.ndarray
            Standard deviation of the input signal.
        """
        if level is None:
            return np.ones((self.nlevel,))
        if level > self.nlevel:
            level = self.nlevel - 1
        if level == self.nlevel:
            return np.array(
                [1.4825 * np.median(np.abs(signal[i])) for i in range(self.nlevel)]
            )
        tmp_sigma = 1.4825 * np.median(np.abs(signal[self.nlevel - 1]))
        return np.array([tmp_sigma for _ in range(self.nlevel)])

    def wav_transform(self, signal):
        """
        Perform wavelet multilevel decomposition on the input signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).

        Returns
        -------
        list
            Wavelet coefficients.
        """
        size = nearest_even_integer(signal.shape[0])
        if self.nlevel == 0:
            self.nlevel = dwt_max_level(
                signal.shape[0], filter_len=self.filter_.dec_len
            )
        return wavedec(signal[:size], self.filter_, level=self.nlevel)

    def denoise(self, signal, coeffs):
        """
        Denoise the input signal based on its wavelet coefficients.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).
        coeffs : list
            Wavelet coefficients.

        Returns
        -------
        np.ndarray
            Denoised signal.
        """
        sigma = self.std(coeffs[1:], level=self.selected_level)
        thr = [
            self.determine_threshold(coeffs[1 + level] / sigma[level], self.energy_perc)
            * sigma[level]
            for level in range(self.nlevel)
        ]
        coeffs[1:] = [
            threshold(c, value=thr[i], mode=self.thr_mode)
            for i, c in enumerate(coeffs[1:])
        ]
        denoised_signal = waverec(coeffs, self.filter_, mode=self.recon_mode)
        if self.normalize:
            denoised_signal = self.scaler.inverse_transform(
                denoised_signal.reshape(-1, 1)
            )[:, 0]
        return denoised_signal

    def determine_threshold(self, signal, energy_perc=0.9):
        """
        Determine the value of the threshold.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).
        energy_perc : float, optional
            Energy retained percentage. Default is 0.9.

        Returns
        -------
        float
            Threshold value.
        """
        if self.method == "universal":
            return self.universal_threshold(signal)
        if self.method == "sqtwolog":
            return self.universal_threshold(signal, sigma=False)
        if self.method == "stein":
            return self.stein_threshold(signal)
        if self.method == "heurstein":
            return self.heur_stein_threshold(signal)
        if self.method == "energy":
            return self.energy_threshold(signal, perc=energy_perc)
        raise ValueError(
            "No such method detected! Set back to default (universal thresholding)!"
        )

    def universal_threshold(self, signal, sigma=True):
        """
        Universal threshold.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).
        sigma : bool, optional
            If True, multiplies the term sqrt(2*log(m)) with the MAD value of the input signal. Default is True.

        Returns
        -------
        float
            Threshold value.
        """
        m = signal.shape[0]
        sd = mad(signal) if sigma else 1.0
        return sd * np.sqrt(2 * np.log(m))

    def stein_threshold(self, signal):
        """
        Stein's unbiased risk estimator.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).

        Returns
        -------
        float
            Threshold value.
        """
        m = signal.shape[0]
        sorted_signal = np.sort(np.abs(signal)) ** 2
        c = np.linspace(m - 1, 0, m)
        s = np.cumsum(sorted_signal) + c * sorted_signal
        risk = (m - (2.0 * np.arange(m)) + s) / m
        ibest = np.argmin(risk)
        return np.sqrt(sorted_signal[ibest])

    def heur_stein_threshold(self, signal):
        """
        Heuristic implementation of Stein's unbiased risk estimator.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).

        Returns
        -------
        float
            Threshold value.
        """
        m, j = dyadic_length(signal)
        magic = np.sqrt(2 * np.log(m))
        eta = (np.linalg.norm(signal) ** 2 - m) / m
        critical = j**1.5 / np.sqrt(m)
        if eta < critical:
            return magic
        return np.min((self.stein_threshold(signal), magic))

    def energy_threshold(self, signal, perc=0.1):
        """
        Energy-based threshold method.

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1D).
        perc : float, optional
            Energy retained percentage. Default is 0.1.

        Returns
        -------
        float
            Threshold value.
        """
        tmp_signal = np.sort(np.abs(signal))[::-1]
        energy_thr = perc * energy(tmp_signal)
        energy_tmp = 0
        for sig in tmp_signal:
            energy_tmp += sig**2
            if energy_tmp >= energy_thr:
                return sig
        return 0.0
