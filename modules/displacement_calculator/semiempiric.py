import os
import sys
import numpy as np
from dataclasses import dataclass, field
from scipy.stats import norm

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from libs.utils.logger_config import get_logger
from libs.utils.config_variables import (
    COEFF_STRUCT_PERIOD,
    BRAY_ZERO_DISPL_PROB_COEFFS,
    BRAY_DISP_AVG_COEFFS,
)

logger = get_logger("displacement_calculator")


@dataclass
class BaseCalculator:
    """
    Base class for displacement calculation.

    Parameters
    ----------
    fail_surf_type : str
        Type of failure surface.
    h_slide_block : float
        Height of the sliding block.
    vs_prom_fail_surf : float
        Average shear wave velocity of the failure surface.
    yield_acc : float
        Yield acceleration.
    spc_acc : list
        Spectral accelerations.
    periods : list
        Periods corresponding to the spectral accelerations.
    magnitude : float
        Earthquake magnitude.
    desv_std : float
        Standard deviation.
    """

    fail_surf_type: str
    h_slide_block: float
    vs_prom_fail_surf: float
    yield_acc: float
    spc_acc: list
    periods: list
    magnitude: float
    desv_std: float

    spc_acc_array: np.ndarray = field(init=False)
    periods_array: np.ndarray = field(init=False)
    struct_period: float = field(init=False)
    spc_acc_degr: float = field(init=False)

    def __post_init__(self):
        """
        Post-initialization to calculate additional attributes.
        """
        logger.debug("Initializing BaseCalculator with parameters: %s", self)
        self.spc_acc_array = np.array(self.spc_acc)
        self.periods_array = np.array(self.periods)
        self.struct_period = self.calc_struct_period()
        self.spc_acc_degr = self.calc_degraded_acceleration()
        logger.debug("Calculated struct_period: %f", self.struct_period)
        logger.debug("Calculated spc_acc_degr: %f", self.spc_acc_degr)

    def calc_coeff_struct_period(self):
        """
        Calculate the coefficient for the structural period.

        Returns
        -------
        float
            Coefficient for the structural period.
        """
        return COEFF_STRUCT_PERIOD.get(self.fail_surf_type, 2.6)

    def calc_struct_period(self):
        """
        Calculate the structural period.

        Returns
        -------
        float
            Structural period.
        """
        coeff_ts = self.calc_coeff_struct_period()
        return coeff_ts * self.h_slide_block / self.vs_prom_fail_surf

    def calc_degraded_acceleration(self):
        """
        Calculate the degraded spectral acceleration.

        Returns
        -------
        float
            Degraded spectral acceleration.
        """
        return np.interp(self.struct_period, self.periods_array, self.spc_acc_array)


@dataclass
class Bray2018Calculator(BaseCalculator):
    """
    Bray et al. (2018) displacement calculator.

    Inherits from BaseCalculator.
    """

    def calc_zero_displacement_prob(self):
        """
        Calculate the probability of zero displacement.

        Returns
        -------
        float
            Probability of zero displacement.
        """
        logger.debug("Calculating zero displacement probability")
        a, b, c, d, e, f = BRAY_ZERO_DISPL_PROB_COEFFS
        log_yield_acc = np.log(self.yield_acc)

        prob = 1 - norm.cdf(
            a
            + b * log_yield_acc
            + c * log_yield_acc**2
            + d * self.struct_period * log_yield_acc
            + e * self.struct_period
            + f * np.log(self.spc_acc_degr)
        )
        logger.debug("Calculated zero displacement probability: %f", prob)
        return prob

    def calc_displacement(self):
        """
        Calculate the displacement.

        Returns
        -------
        float
            Calculated displacement.
        """
        logger.debug("Calculating displacement")
        a1, a2, a3, a, b, c, d, e, f = BRAY_DISP_AVG_COEFFS
        log_yield_acc = np.log(self.yield_acc)
        log_spc_acc_degr = np.log(self.spc_acc_degr)

        displacement = np.exp(
            a1
            + a * log_yield_acc
            + b * log_yield_acc**2
            + c * log_yield_acc * log_spc_acc_degr
            + d * log_spc_acc_degr
            + e * log_spc_acc_degr**2
            + a2 * self.struct_period
            + a3 * self.struct_period**2
            + f * self.magnitude
            + self.desv_std
        )
        logger.debug("Calculated displacement: %f", displacement)
        return displacement


class DisplacementCalculator:
    """
    Displacement calculator class.

    Parameters
    ----------
    method : str
        Method to use for displacement calculation.
    """

    def __init__(self, method: str):
        """
        Initialize the DisplacementCalculator class.

        Parameters
        ----------
        method : str
            Method to use for displacement calculation.
        """
        logger.debug("Initializing DisplacementCalculator with method: %s", method)
        self.method = method
        self.displacement = None

    def execute(self, **kwargs):
        """
        Execute the displacement calculation.

        Parameters
        ----------
        **kwargs : dict
            Parameters for the calculation.
        """
        logger.debug("Executing displacement calculation with parameters: %s", kwargs)
        try:
            # Verify if the class exists in the current context
            calculator_class = globals().get(self.method)
            if not calculator_class or not issubclass(calculator_class, BaseCalculator):
                logger.error("Method '%s' is not a valid Calculator class", self.method)
                raise ValueError(
                    f"Method '{self.method}' is not a valid Calculator class"
                )

            calculator = calculator_class(**kwargs)
            self.displacement = calculator.calc_displacement()
            logger.debug("Displacement calculation result: %f", self.displacement)
        except Exception as e:
            logger.error("Error during displacement calculation: %s", e)
            raise

    def get_result(self):
        """
        Get the displacement result.

        Returns
        -------
        float
            Displacement result.
        """
        logger.debug("Getting displacement result")
        if self.displacement is None:
            logger.warning(
                "Displacement result is not available. Execute the calculation first."
            )
        return self.displacement
