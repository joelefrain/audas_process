import numba
import numpy as np
import numpy.typing as npt
import pykooh
from scipy.optimize import minimize

from .motion import GRAVITY, Motion, WaveField
from .site import Layer, Location, Profile


class AbstractCalculator:
    def __init__(self):
        self._loc_input: None | Location = None
        self._motion: None | Motion = None
        self._profile: None | Profile = None

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
        self._motion = motion
        self._profile = profile
        self._loc_input = loc_input

        if reset_layers:
            # Set initial properties
            for layer in profile:
                layer.reset()
                if layer.strain is None:
                    layer.strain = 0.0

    @property
    def motion(self):
        return self._motion

    @property
    def profile(self):
        return self._profile

    @property
    def loc_input(self):
        return self._loc_input


@numba.jit(nopython=True)
def my_trapz(thickness, prop, depth_max):
    total = 0
    depth = 0

    for t, p in zip(thickness, prop):
        depth += t
        if depth_max < depth:
            # Partial layer
            total += (t - (depth - depth_max)) * p
            break
        total += t * p
    else:
        # Final infinite layer
        total += (depth_max - depth) * p

    return total / depth_max


class QuarterWaveLenCalculator(AbstractCalculator):
    """Compute quarter-wave length site amplification.

    No consideration for nolninearity is made by this calculator.
    """

    name = "QWL"

    def __init__(self, site_atten=None, method="standard"):
        super().__init__()
        self._site_atten = site_atten
        self._method = method

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        loc_input: :class:`~.base.site.Location`
            Location of the input motion.
        """
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

        self._crustal_amp, self._site_term = self._calc_amp(
            profile.density, profile.thickness, profile.slowness
        )

    @staticmethod
    def correction_ba23(x):
        a = 0.560
        b = -1.301
        s = 1.398
        d = 4.000
        e = 6.000
        g = 2.000
        h = 0.760
        p = 3.000
        q = 0.333

        fact = (x - b) / s

        eta = (a * fact**d) / ((1 - fact**e) ** g + h * fact**p) ** q

        return eta

    @property
    def method(self) -> str:
        return self._method

    @property
    def crustal_amp(self) -> np.ndarray:
        return self._crustal_amp

    @property
    def site_term(self) -> np.ndarray:
        return self._site_term

    @property
    def site_atten(self) -> float | None:
        return self._site_atten

    def _calc_amp(
        self, density: npt.ArrayLike, thickness: npt.ArrayLike, slowness: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs = self.motion.freqs
        # 1/4 wavelength depth -- estimated for mean slowness
        qwl_depth = 1 / (4 * np.mean(slowness) * freqs)

        def qwl_average(param):
            return np.array([my_trapz(thickness, param, qd) for qd in qwl_depth])

        for _ in range(50):
            qwl_slowness = qwl_average(slowness)
            prev_qwl_depth = qwl_depth

            # Compute the mean between the previous depths and the newly
            # computed depths. If the new value is just taken, then this
            # algorithm can osccilate between two solutions.
            qwl_depth = np.mean(
                np.c_[prev_qwl_depth, 1 / (4 * qwl_slowness * freqs)], axis=1
            )
            if np.allclose(prev_qwl_depth, qwl_depth, rtol=0.01):
                break
        else:
            raise RuntimeError("QWL calcuation did not converge.")

        qwl_density = qwl_average(density)

        if self.method == "standard":
            eta = 0.5
        elif self.method == "ba23":
            total_depth = np.sum(thickness[:-1])
            total_slow = my_trapz(thickness, slowness, total_depth)
            freq_bot = 1.0 / (4 * total_depth * total_slow)
            eta = self.correction_ba23(np.log10(freqs / freq_bot))
        else:
            raise NotImplementedError

        crustal_amp = (
            (density[-1] / slowness[-1]) / (qwl_density / qwl_slowness)
        ) ** eta

        site_term = np.array(crustal_amp)
        if self.site_atten:
            site_term *= np.exp(-np.pi * self.site_atten * freqs)

        return crustal_amp, site_term

    def fit(
        self,
        target_type,
        target,
        adjust_thickness=False,
        adjust_site_atten=False,
        adjust_source_vel=False,
    ):
        """
        Fit to a target crustal amplification or site term.

        The fitting process adjusts the velocity, site attenuation, and layer
        thickness (if enabled) to fit a target values. The frequency range is
        specified by the input motion.

        Parameters
        ----------
        target_type: str
            Options are 'crustal_amp' to only fit to the crustal amplification,
             or 'site_term' to fit both the velocity and the site attenuation
             parameter.
        target: `array_like`
            Target values.
        adjust_thickness: bool (optional)
            If the thickness of the layers is adjusted as well, default: False.
        adjust_site_atten: bool (optional)
            If the site attenuation is adjusted as well, default: False.
        adjust_source_vel: bool (optional)
            If the source velocity should be adjusted, default: False.
        Returns
        -------
        profile: `pyrsa.site.Profile`
            profile optimized to fit a target amplification.
        """
        density = self.profile.density

        nl = len(density)

        # Slowness bounds
        slowness = self.profile.slowness
        thickness = self.profile.thickness
        site_atten = self._site_atten

        # Slowness
        initial = slowness
        bounds = 1 / np.tile((4000, 100), (nl, 1))
        if not adjust_source_vel:
            bounds[-1] = (initial[-1], initial[-1])

        # Thickness bounds
        if adjust_thickness:
            bounds = np.r_[bounds, [[t / 2, 2 * t] for t in thickness]]
            initial = np.r_[initial, thickness]

        # Site attenuation bounds
        if adjust_site_atten:
            bounds = np.r_[bounds, [[0.0001, 0.200]]]
            initial = np.r_[initial, self.site_atten]

        def calc_rmse(this, that):
            return np.mean(((this - that) / that) ** 2)

        def err(x):
            _slowness = x[0:nl]
            if adjust_thickness:
                _thickness = x[nl : (2 * nl)]
            else:
                _thickness = thickness
            if adjust_site_atten:
                self._site_atten = x[-1]

            crustal_amp, site_term = self._calc_amp(density, _thickness, _slowness)

            calc = crustal_amp if target_type == "crustal_amp" else site_term

            err = 10 * calc_rmse(target, calc)
            # Prefer the original values so add the difference to the error
            err += calc_rmse(slowness, _slowness)
            if adjust_thickness:
                err += calc_rmse(thickness, _thickness)
            if adjust_site_atten:
                err += calc_rmse(self._site_atten, site_atten)
            return err

        res = minimize(err, initial, method="L-BFGS-B", bounds=bounds)

        slowness = res.x[0:nl]
        if adjust_thickness:
            thickness = res.x[nl : (2 * nl)]

        profile = Profile(
            [
                Layer(layer.soil_type, thick, 1 / slow)
                for layer, thick, slow in zip(self.profile, thickness, slowness)
            ],
            self.profile.wt_depth,
        )
        # Update the calculated amplificaiton
        self(self.motion, profile, self.loc_input)


class LinearElasticCalculator(AbstractCalculator):
    """Class for performing linear elastic site response."""

    name = "LE"

    def __init__(self):
        super().__init__()

        self._waves_a = np.array([])
        self._waves_b = np.array([])
        self._wave_nums = np.array([])

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        loc_input: :class:`~.base.site.Location`
            Location of the input motion.
        """
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

        self._calc_waves(motion.angular_freqs, profile)

    def _calc_waves(self, angular_freqs, profile):
        """Compute the wave numbers and amplitudes (up- and down-going).

        Parameters
        ----------
        angular_freqs: :class:`numpy.ndarray`
            Angular frequency at which the waves are computed.

        profile: :class:`~.base.site.Profile`
            Site profile.
        """

        # Compute the complex wave numbers of the system
        wave_nums = np.empty((len(profile), len(angular_freqs)), complex)
        for i, layer in enumerate(profile):
            wave_nums[i, :] = angular_freqs / layer.comp_shear_vel

        # Compute the waves. In the top surface layer, the up-going and
        # down-going waves have an amplitude of 1 as they are completely
        # reflected at the surface.
        waves_a = np.ones_like(wave_nums, complex)
        waves_b = np.ones_like(wave_nums, complex)
        for i, layer in enumerate(profile[:-1]):
            # Complex impedance -- wave number can be zero which causes an
            # error.
            with np.errstate(invalid="ignore"):
                cimped = (wave_nums[i] * layer.comp_shear_mod) / (
                    wave_nums[i + 1] * profile[i + 1].comp_shear_mod
                )

            # Complex term to simplify equations -- uses full layer height
            cterm = 1j * wave_nums[i, :] * layer.thickness

            waves_a[i + 1, :] = 0.5 * waves_a[i] * (1 + cimped) * np.exp(
                cterm
            ) + 0.5 * waves_b[i] * (1 - cimped) * np.exp(-cterm)
            waves_b[i + 1, :] = 0.5 * waves_a[i] * (1 - cimped) * np.exp(
                cterm
            ) + 0.5 * waves_b[i] * (1 + cimped) * np.exp(-cterm)

            # Set wave amplitudes with zero frequency to 1
            mask = ~np.isfinite(cimped)
            waves_a[i + 1, mask] = 1.0
            waves_b[i + 1, mask] = 1.0

        # fixme: Better way to handle this?
        # Set wave amplitudes to 1 at frequencies near 0
        mask = np.isclose(angular_freqs, 0)
        waves_a[-1, mask] = 1.0
        waves_b[-1, mask] = 1.0

        self._waves_a = waves_a
        self._waves_b = waves_b
        self._wave_nums = wave_nums

    def wave_at_location(self, loc: Location) -> np.ndarray:
        """Compute the wave field at specific location.

        Parameters
        ----------
        loc : site.Location
            :class:`site.Location` of the input

        Returns
        -------
        `np.ndarray`
            Amplitude and phase of waves
        """
        cterm = 1j * self._wave_nums[loc.index] * loc.depth_within

        if loc.wave_field == WaveField.within:
            return self._waves_a[loc.index] * np.exp(cterm) + self._waves_b[
                loc.index
            ] * np.exp(-cterm)
        elif loc.wave_field == WaveField.outcrop:
            return 2 * self._waves_a[loc.index] * np.exp(cterm)
        elif loc.wave_field == WaveField.incoming_only:
            return self._waves_a[loc.index] * np.exp(cterm)
        else:
            raise NotImplementedError

    def calc_accel_tf(self, lin, lout):
        """Compute the acceleration transfer function.

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        """
        tf = self.wave_at_location(lout) / self.wave_at_location(lin)
        return tf

    def calc_stress_tf(self, lin, lout, damped):
        """Compute the stress transfer function.

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        """
        tf = self.calc_strain_tf(lin, lout)
        if damped:
            # Scale by complex shear modulus to include the influence of
            # damping
            tf *= lout.layer.comp_shear_mod
        else:
            tf *= lout.layer.shear_mod

        return tf

    def calc_strain_tf(self, lin, lout):
        """Compute the strain transfer function from `lout` to
        `location_in`.

        The strain transfer function from the acceleration at layer `n`
        (outcrop) to the mid-height of layer `m` (within) is defined as

        Parameters
        ----------
        lin : :class:`~site.Location`
            Location of input
        lout : :class:`~site.Location`
            Location of output. Note that this would typically be midheight
            of the layer.

        Returns
        -------
        strain_tf : :class:`numpy.ndarray`
            Transfer function to be applied to an acceleration FAS.
        """
        # FIXME: Correct discussion for using acceleration FAS
        # Strain(angFreq, z=h_m/2)
        # ------------------------ =
        #    accel_n(angFreq)
        #
        #          i k*_m [ A_m exp(i k*_m h_m / 2) - B_m exp(-i k*_m h_m / 2)]
        #          ------------------------------------------------------------
        #                         -angFreq^2 (2 * A_n)
        #
        assert lout.wave_field == WaveField.within

        ang_freqs = self.motion.angular_freqs
        # The numerator cannot be computed using wave_at_location() because
        # it is A - B.
        cterm = 1j * self._wave_nums[lout.index, :] * lout.depth_within
        numer = (
            1j
            * self._wave_nums[lout.index, :]
            * (
                self._waves_a[lout.index, :] * np.exp(cterm)
                - self._waves_b[lout.index, :] * np.exp(-cterm)
            )
        )
        denom = -(ang_freqs**2) * self.wave_at_location(lin)

        # Only compute transfer function for non-zero frequencies
        mask = ~np.isclose(ang_freqs, 0)
        tf = np.zeros_like(mask, dtype=complex)
        # Scale into units from gravity
        tf[mask] = GRAVITY * numer[mask] / denom[mask]

        return tf


class EquivalentLinearCalculator(LinearElasticCalculator):
    """Class for performing equivalent-linear elastic site response."""

    name = "EQL"

    def __init__(
        self, strain_ratio=0.65, tolerance=0.01, max_iterations=15, strain_limit=0.05
    ):
        """Initialize the class.

        Parameters
        ----------
        strain_ratio: float, default=0.65
            Ratio between the maximum strain and effective strain used to
            compute strain compatible properties.

        tolerance: float, default=0.01
            Tolerance in the iterative properties, which would cause the
            iterative process to terminate.

        max_iterations: int, default=15
            Maximum number of iterations to perform.

        strain_limit: float, default=0.05
            Limit of strain in calculations. If this strain is exceed, the
            iterative calculation is ended.
        """
        super().__init__()
        self._strain_ratio = strain_ratio
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._strain_limit = strain_limit

    def __call__(
        self,
        motion: Motion,
        profile: Profile,
        loc_input: Location,
        reset_layers=True,
        **kwds,
    ):
        """Perform the wave propagation.

        Parameters
        ----------
        motion: :class:`~.base.motion.Motion`
            Input motion.

        profile: :class:`~.base.site.Profile`
            Site profile.

        loc_input: :class:`~.base.site.Location`
            Location of the input motion.
        """
        super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)

        if reset_layers:
            # Use the previously established layer strains
            self._estimate_strains()

        iteration = 0
        # The iteration at which strains were last limited
        limited_iter = -2
        limited_strains = False

        while iteration < self.max_iterations:
            limited_strains = False
            self._calc_waves(motion.angular_freqs, profile)

            for index, layer in enumerate(profile[:-1]):
                loc_layer = Location(index, layer, "within", layer.thickness / 2)

                # Compute the representative strain(s) within the layer. FDM
                #  will provide a vector of strains.
                strain = self._calc_strain(loc_input, loc_layer, motion)
                if self._strain_limit and np.any(strain > self._strain_limit):
                    limited_strains = True
                    strain = np.minimum(strain, self._strain_limit)
                layer.strain = strain

            # Maximum error (damping and shear modulus) over all layers
            max_error = max(profile.max_error)
            if max_error < self.tolerance:
                break

            # Break, if the strains were limited the last two iterations.
            if limited_strains:
                if limited_iter == (iteration - 1):
                    raise RuntimeError("Strain limit exceeded.")
                else:
                    limited_iter = iteration

            iteration += 1

        # Compute the maximum strain within the profile.
        for index, layer in enumerate(profile[:-1]):
            loc_layer = Location(index, layer, "within", layer.thickness / 2)
            layer.strain_max = self._calc_strain_max(loc_input, loc_layer, motion)

    def _estimate_strains(self):
        """Compute an estimate of the strains."""
        # Estimate the strain based on the PGV and shear-wave velocity
        for layer in self._profile:
            layer.reset()
            # PGV in units of cm/sec
            layer.strain = (self._motion.pgv / 100) / layer.initial_shear_vel

    @property
    def strain_ratio(self):
        return self._strain_ratio

    @property
    def tolerance(self):
        return self._tolerance

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def strain_limit(self):
        return self._strain_limit

    @classmethod
    def calc_strain_ratio(cls, mag):
        """Compute the effective strain ratio using Idriss and Sun (1992).

        Parameters
        ----------
        mag: float
            Magnitude of the input motion.

        Returns
        -------
        strain_ratio : float
            Effective strain ratio

        References
        ----------
        .. [1] Idriss, I. M., & Sun, J. I. (1992). SHAKE91: A computer program
            for conducting equivalent linear seismic response analyses of
            horizontally layered soil deposits. Center for Geotechnical
            Modeling, Department of Civil and Environmental Engineering,
            University of California, Davis, CA.
        """
        return (mag - 1) / 10

    def _calc_strain(self, loc_input, loc_layer, motion, *args):
        """Compute the strain used for iterations of material properties."""
        strain_max = self._calc_strain_max(loc_input, loc_layer, motion, *args)
        return self.strain_ratio * strain_max

    def _calc_strain_max(self, loc_input, loc_layer, motion, *args):
        """Compute the effective strain at the center of a layer."""
        return motion.calc_peak(self.calc_strain_tf(loc_input, loc_layer))


class FrequencyDependentEqlCalculator(EquivalentLinearCalculator):
    """Class for performing equivalent-linear elastic site response with
    frequency-dependent modulii and damping.

    Parameters
    ----------
    method: str
        method for computing the strain spectrum:
         - raw: no modification to the strain spectrum.
         - ka02: use the Kausel & Assimaki (2002) defined shape for a  smooth spectrum
           for the strain.
         - ko:##: use Konno-Omachi with a bandwith of ##.
    strain_ratio: float, default=1.00
        ratio between the maximum strain and effective strain used to compute
        strain compatible properties. There is not clear guidance the use of
        the effective strain ratio. However, given the nature of the method,
        it would make sense not to include the an effective strain ratio.
    tolerance: float, default=0.01
        tolerance in the iterative properties, which would cause the iterative
        process to terminate.
    max_iterations: int, default=15
        maximum number of iterations to perform.

    strain_limit: float, default=0.05
        Limit of strain in calculations. If this strain is exceed, the
        iterative calculation is ended.

    References
    ----------
    .. [1] Kausel, E., & Assimaki, D. (2002). Seismic simulation of inelastic
        soils via frequency-dependent moduli and damping. Journal of
        Engineering Mechanics, 128(1), 34-47.
    """

    name = "FDM-KA"

    def __init__(
        self,
        method: str = "raw",
        strain_ratio: float = 1.0,
        tolerance: float = 0.01,
        max_iterations: int = 15,
        strain_limit: float = 0.05,
    ):
        """Initialize the class."""
        super().__init__(strain_ratio, tolerance, max_iterations, strain_limit)

        self._method = method
        self._smoother = None

    @property
    def method(self):
        return self._method

    def _estimate_strains(self):
        """Estimate the strains by running an EQL site response.

        This step was recommended in Section 8.3.1 of Zalachoris (2014).
        """
        eql = EquivalentLinearCalculator()
        eql(self._motion, self._profile, self._loc_input)

    def _calc_strain(self, loc_input, loc_layer, motion, *args):
        freqs = np.array(motion.freqs)
        strain_tf = self.calc_strain_tf(loc_input, loc_layer)
        strain_fas = np.abs(strain_tf * motion.fourier_amps)
        # Maximum strain in the time domain modified by the effective strain
        # ratio
        strain_eff = self.strain_ratio * motion.calc_peak(strain_tf)

        if self._method == "ka02":
            # Equation (8)
            freq_avg = np.trapz(freqs * strain_fas, x=freqs) / np.trapz(
                strain_fas, x=freqs
            )

            # Find the average strain at frequencies less than the average
            # frequency
            # Equation (8)
            mask = freqs < freq_avg
            strain_avg = np.trapz(strain_fas[mask], x=freqs[mask]) / freq_avg

            # Normalize the frequency and strain by the average values
            freqs /= freq_avg
            strain_fas /= strain_avg

            # Fit the smoothed model at frequencies greater than the average
            A = np.c_[-freqs[~mask], -np.log(freqs[~mask])]
            a, b = np.linalg.lstsq(A, np.log(strain_fas[~mask]), rcond=None)[0]
            # This is a modification of the published method that ensures a
            # smooth transition in the strain. Make sure the frequencies are zero.
            shape = np.minimum(
                1,
                np.exp(-a * freqs)
                / np.maximum(np.finfo(float).eps, np.power(freqs, b)),
            )
            strains = strain_eff * shape
        elif self._method.startswith("ko:"):
            if self._smoother is None or not self._smoother.freqs_match(motion.freqs):
                bandwidth = float(self._method[3:])
                self._smoother = pykooh.CachedSmoother(
                    motion.freqs, motion.freqs, bandwidth=bandwidth, normalize=True
                )

            # Konno-Omachi smoothing
            # strain_fas_sm = pykooh.smooth(freqs, freqs, strain_fas, self._bandwidth)
            strains = strain_eff * self._smoother(strain_fas) / np.max(strain_fas)

        else:
            strains = strain_eff * strain_fas / np.max(strain_fas)

        return strains

# # DEBUGME: 
# class NonLinearCalculator(AbstractCalculator):
#     """Class for performing non-linear site response analysis."""
    
#     name = "NL"

#     def __init__(self, model_params=None):
#         """
#         Initialize the class.

#         Parameters
#         ----------
#         model_params: dict, optional
#             Dictionary containing parameters for the non-linear constitutive model.
#             Default is None, meaning default parameters will be used.
#         """
#         super().__init__()
#         self.model_params = model_params or {
#             "alpha": 0.8,  # non-linearity parameter
#             "beta": 0.3,   # curvature parameter in the non-linear model
#         }
#         self.strains = []

#     def __call__(self, motion: Motion, profile: Profile, loc_input: Location, reset_layers=True, **kwds):
#         """
#         Perform the wave propagation with non-linear response.

#         Parameters
#         ----------
#         motion: :class:~.base.motion.Motion
#             Input motion.
        
#         profile: :class:~.base.site.Profile
#             Site profile.
        
#         loc_input: :class:~.base.site.Location
#             Location of the input motion.
#         """
#         super().__call__(motion, profile, loc_input, reset_layers=reset_layers, **kwds)
        
#         # Calculate wave propagation for each layer considering non-linearity
#         self.strains = self._calc_non_linear_strain(profile, motion.angular_freqs)

#     def _calc_non_linear_strain(self, profile, angular_freqs):
#         """
#         Compute strain in each layer with a non-linear constitutive model.
        
#         Parameters
#         ----------
#         profile: :class:~.base.site.Profile
#             Site profile.
        
#         angular_freqs: numpy.ndarray
#             Angular frequencies from the motion.
        
#         Returns
#         -------
#         strains: list of numpy.ndarray
#             Computed strains for each layer in profile.
#         """
#         strains = []
#         for i, layer in enumerate(profile):
#             # Non-linear strain computation
#             strain = self._non_linear_model(layer, angular_freqs)
#             strains.append(strain)
#             layer.strain = strain
#         return strains

#     def _non_linear_model(self, layer, angular_freqs):
#         """
#         Compute strain based on non-linear constitutive model (e.g., Ramberg-Osgood).
        
#         Parameters
#         ----------
#         layer: Layer
#             Geologic layer being analyzed.
        
#         angular_freqs: numpy.ndarray
#             Angular frequencies to compute wave interactions.
        
#         Returns
#         -------
#         strain: numpy.ndarray
#             Strain values computed with non-linear model for the layer.
#         """
#         shear_mod = layer.initial_shear_mod
#         alpha = self.model_params["alpha"]
#         beta = self.model_params["beta"]

#         # Non-linear model for strain as a function of angular frequency
#         strain = (alpha * angular_freqs) / (1 + beta * (angular_freqs / shear_mod) ** 2)
#         return strain

#     def calc_accel_tf(self, lin, lout):
#         """
#         Calculate the acceleration transfer function for non-linear response.

#         Parameters
#         ----------
#         lin: Location
#             Location of input.
        
#         lout: Location
#             Location of output.
        
#         Returns
#         -------
#         tf: numpy.ndarray
#             Transfer function for acceleration between input and output locations.
#         """
#         input_wave = self.wave_at_location(lin)
#         output_wave = self.wave_at_location(lout)
        
#         # Non-linear modification to acceleration transfer function
#         tf = np.abs(output_wave / input_wave)
#         return tf

#     def calc_stress_tf(self, lin, lout):
#         """
#         Calculate stress transfer function with non-linear model.
        
#         Parameters
#         ----------
#         lin: Location
#             Location of input.
        
#         lout: Location
#             Location of output.
        
#         Returns
#         -------
#         tf: numpy.ndarray
#             Stress transfer function.
#         """
#         tf_strain = self.calc_strain_tf(lin, lout)
#         tf_stress = tf_strain * lout.layer.non_linear_shear_mod  # Non-linear shear modulus
#         return tf_stress

#     def calc_strain_tf(self, lin, lout):
#         """
#         Calculate strain transfer function with non-linear properties.

#         Parameters
#         ----------
#         lin: Location
#             Location of input.
        
#         lout: Location
#             Location of output.
        
#         Returns
#         -------
#         strain_tf: numpy.ndarray
#             Strain transfer function.
#         """
#         ang_freqs = self.motion.angular_freqs
#         # Non-linear strain as function of angular frequencies and location properties
#         strain_tf = self._non_linear_model(lout.layer, ang_freqs) / self._non_linear_model(lin.layer, ang_freqs)
#         return strain_tf
