import enum
import re

import numpy as np
import pyrvt

# Gravity in m/sec²
from scipy.constants import g as GRAVITY


class WaveField(enum.Enum):
    outcrop = 0
    within = 1
    incoming_only = 2


class Motion:
    def __init__(self, freqs=None):
        object.__init__(self)

        self._freqs = np.array([] if freqs is None else freqs)
        self._pga = None
        self._pgv = None

    @property
    def freqs(self):
        return self._freqs

    @property
    def angular_freqs(self):
        return 2 * np.pi * self.freqs

    @property
    def pgv(self):
        """Peak ground velocity [cm/sec]."""
        if self._pgv is None:
            self._pgv = self.calc_pgv()
        return self._pgv

    @property
    def pga(self):
        """Peak ground acceleration [g]"""
        if self._pga is None:
            self._pga = self.calc_pga()
        return self._pga

    def calc_pgv(self, tf=None):
        tf = 1 if tf is None else np.asarray(tf)
        # Compute transfer function from acceleration to velocity
        # only over non-zero frequencies
        mask = ~np.isclose(self.angular_freqs, 0)
        tf_av = np.zeros_like(mask, dtype=complex)
        tf_av[mask] = 1 / (self.angular_freqs[mask] * 1j)

        pgv = GRAVITY * 100 * self.calc_peak(tf_av * tf)
        return pgv

    def calc_pga(self, tf=None):
        tf = 1 if tf is None else np.asarray(tf)
        return self.calc_peak(tf)

    def calc_peak(self, tf=None, **kwargs):
        raise NotImplementedError


class TimeSeriesMotion(Motion):
    """Time-series motion for time series based site response analysis."""

    def __init__(
        self, filename: str, description: str, time_step: float, accels, fa_length=None
    ):
        """Initialize the class from specified acceleration values.

        The *filename* and *description* parameters are only used to help track the
        motion.

        Parameters
        ----------
        filename: str
            Source of data
        description: str
            Description to store helpful information
        time_step: float
            Time step of the accleration values
        accels: array_like
            Accelerations in units of *g*
        fa_length: optional int
            Length to use for the Fourier amplitude spectrum. If not provided, will be
            automatically computed to the next power of 2.
        """
        Motion.__init__(self)

        self._filename = filename
        self._description = description
        self._time_step = time_step
        self._accels = np.asarray(accels)

        self._calc_fourier_spectrum(fa_length)

    @property
    def accels(self):
        return self._accels

    @property
    def filename(self):
        return self._filename

    @property
    def description(self):
        return self._description

    @property
    def time_step(self):
        return self._time_step

    @property
    def times(self):
        return self._time_step * np.arange(self._accels.size)

    @property
    def freqs(self):
        """Return the frequencies."""
        if self._freqs is None:
            self._calc_fourier_spectrum()

        return self._freqs

    @property
    def fourier_amps(self):
        """Return the frequencies."""
        if self._fourier_amps is None:
            self._calc_fourier_spectrum()

        # Normalize the Fourier amplitude by the time step
        return self.time_step * self._fourier_amps

    def calc_time_series(self, tf=None):
        if tf is None:
            ts = np.fft.irfft(self.fourier_amps / self.time_step)
        else:
            ts = np.fft.irfft(tf * self.fourier_amps / self.time_step)
        return ts

    def calc_peak(self, tf=None, **kwargs):
        ts = self.calc_time_series(tf)
        return np.abs(ts).max()

    def calc_osc_accels(self, osc_freqs, osc_damping=0.05, tf=None):
        """Compute the pseudo-acceleration spectral response of an oscillator
        with a specific frequency and damping.

        Parameters
        ----------
        osc_freq : float
            Frequency of the oscillator (Hz).
        osc_damping : float
            Fractional damping of the oscillator (dec). For example, 0.05 for a
            damping ratio of 5%.
        tf : array_like, optional
            Transfer function to be applied to motion prior calculation of the
            oscillator response.

        Returns
        -------
        spec_accels : :class:`numpy.ndarray`
            Peak pseudo-spectral acceleration of the oscillator
        """
        if tf is None:
            tf = np.ones_like(self.freqs)
        else:
            tf = np.asarray(tf).astype(complex)

        resp = np.array(
            [
                self.calc_peak(tf * self._calc_sdof_tf(of, osc_damping))
                for of in osc_freqs
            ]
        )
        return resp

    def _calc_fourier_spectrum(self, fa_length=None):
        """Compute the Fourier Amplitude Spectrum of the time series."""

        if fa_length is None:
            # Use the next power of 2 for the length
            n = 1
            while n < self.accels.size:
                n <<= 1
        else:
            n = fa_length

        self._fourier_amps = np.fft.rfft(self._accels, n)

        freq_step = 1.0 / (2 * self._time_step * (n / 2))
        self._freqs = freq_step * np.arange(1 + n / 2)

    def _calc_sdof_tf(self, osc_freq, damping=0.05):
        """Compute the transfer function for a single-degree-of-freedom
        oscillator.

        The transfer function computes the pseudo-spectral acceleration.

        Parameters
        ----------
        osc_freq : float
            natural frequency of the oscillator [Hz]
        damping : float, optional
            damping ratio of the oscillator in decimal. Default value is
            0.05, or 5%.

        Returns
        -------
        tf : :class:`numpy.ndarray`
            Complex-valued transfer function with length equal to `self.freq`.
        """
        return -(osc_freq**2.0) / (
            np.square(self.freqs)
            - np.square(osc_freq)
            - 2.0j * damping * osc_freq * self.freqs
        )

    @classmethod
    def load_at2_file(cls, filename, scale=1.0):
        """Read an AT2 formatted time series.

        Parameters
        ----------
        filename: str
            Filename to open.
        scale: float, default: 1.
            Scale factor to apply to the motion.
        """
        with open(filename) as fp:
            next(fp)
            description = next(fp).strip()
            next(fp)
            parts = next(fp).split()
            time_step = float(parts[1])

            accels = np.array([float(part) for line in fp for part in line.split()])

        accels *= scale
        return cls(filename, description, time_step, accels)

    @classmethod
    def load_smc_file(cls, filename, scale=1.0):
        """Read an SMC formatted time series.

        Format of the time series is provided by:
            https://escweb.wr.usgs.gov/nsmp-data/smcfmt.html

        Parameters
        ----------
        filename: str
            Filename to open.
        scale: float, default: 1.
            Scale factor to apply to the motion.
        """
        from .tools import parse_fixed_width

        with open(filename) as fp:
            lines = list(fp)

        # 11 lines of strings
        lines_str = [lines.pop(0) for _ in range(11)]

        if lines_str[0].strip() != "2 CORRECTED ACCELEROGRAM":
            raise RuntimeWarning("Loading uncorrected SMC file.")

        m = re.search("station =(.+)component=(.+)", lines_str[5])
        description = "; ".join([g.strip() for g in m.groups()])

        # 6 lines of (8i10) formatted integers
        values_int = parse_fixed_width(
            48 * [(10, int)], [lines.pop(0) for _ in range(6)]
        )
        count_comment = values_int[15]
        count = values_int[16]

        # 10 lines of (5e15.7) formatted floats
        values_float = parse_fixed_width(
            50 * [(15, float)], [lines.pop(0) for _ in range(10)]
        )
        time_step = 1 / values_float[1]

        # Skip comments
        lines = lines[count_comment:]

        accels = np.array(
            parse_fixed_width(
                count
                * [
                    (10, float),
                ],
                lines,
            )
        )
        accels *= scale

        return TimeSeriesMotion(filename, description, time_step, accels)


# FIXME: How do multiple inheritence properly?
class RvtMotion(pyrvt.motions.RvtMotion, Motion):
    """RVT motion based on user specified Fourier amplitude spectrum and duration."""

    def __init__(
        self, freqs, fourier_amps, duration=None, peak_calculator=None, calc_kwds=None
    ):
        Motion.__init__(self)
        pyrvt.motions.RvtMotion.__init__(
            self,
            np.asarray(freqs),
            np.asarray(fourier_amps),
            duration=duration,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )


class CompatibleRvtMotion(pyrvt.motions.CompatibleRvtMotion, Motion):
    """RVT motion based on user specified acceleration response spectrum and
    duration."""

    def __init__(
        self,
        osc_freqs,
        osc_accels_target,
        duration=None,
        osc_damping=0.05,
        event_kwds=None,
        window_len=None,
        peak_calculator=None,
        calc_kwds=None,
    ):
        Motion.__init__(self)
        pyrvt.motions.CompatibleRvtMotion.__init__(
            self,
            osc_freqs,
            osc_accels_target,
            duration=duration,
            osc_damping=osc_damping,
            event_kwds=event_kwds,
            window_len=window_len,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )


class SourceTheoryRvtMotion(pyrvt.motions.SourceTheoryMotion, Motion):
    """RVT motion based on seismological point source model and earthquake scenario
    parameters."""

    def __init__(
        self,
        magnitude,
        distance,
        region,
        stress_drop=None,
        depth=8,
        peak_calculator=None,
        calc_kwds=None,
    ):
        Motion.__init__(self)
        pyrvt.motions.SourceTheoryMotion.__init__(
            self,
            magnitude,
            distance,
            region,
            stress_drop,
            depth,
            peak_calculator=peak_calculator,
            calc_kwds=calc_kwds,
        )
