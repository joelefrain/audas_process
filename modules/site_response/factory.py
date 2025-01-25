import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import importlib
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import site, propagation, motion, output
from libs.utils.logger_config import get_logger
from libs.utils.config_variables import (
    TYPE_MOTION_DEFAULT,
    LOG_FREQ_PARAMS,
    DAMPING_DEFAULT,
    EARTH_PRES_COEFF_DEFAULT,
    MOTION_COMPONENTS,
    OUTPUT_TYPES_SITE_RESPONSE,
)

module = "site_response"
logger = get_logger(module)

freq_range = np.logspace(*LOG_FREQ_PARAMS)


class SoilFactory:
    """
    Factory to create instances of specific soil types based on input data.
    """

    @staticmethod
    def get_soil(row):
        """
        Returns an instance of the soil type specified in the input row.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        BaseSoilType
            Instance of the appropriate soil type.

        Raises
        ------
        ValueError
            If more than one of the relevant columns contains a value.
        AttributeError
            If the specified class is not found in the `site` module.
        """
        logger.debug("Getting soil type for row: %s", row)
        row = SoilFactory._replace_nan_with_empty(row)
        valid_columns = SoilFactory._get_valid_columns(row)

        if len(valid_columns) != 1:
            raise ValueError(
                "Just one among 'soil_type', 'deg_curves', or 'author_curves' must have a value."
            )

        if row.get("soil_type"):
            return SoilFactory._create_soil_instance(row)
        elif row.get("deg_curves"):
            return SoilFactory._create_soil_from_file(row)
        elif row.get("author_curves"):
            return SoilFactory._create_soil_from_published(row)

    @staticmethod
    def _replace_nan_with_empty(row):
        """
        Replace NaN values with empty strings in the specified columns.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        dict
            Updated dictionary with NaN values replaced by empty strings.
        """
        for col in ["soil_type", "deg_curves", "author_curves"]:
            if pd.isna(row.get(col)):
                row[col] = ""
        return row

    @staticmethod
    def _get_valid_columns(row):
        """
        Get columns with valid (non-empty) values.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        list
            List of columns with valid values.
        """
        valid_columns = [
            col for col in ["soil_type", "deg_curves", "author_curves"] if row.get(col)
        ]
        for col in valid_columns:
            logger.info(f"{col} : {row.get(col)}")
        return valid_columns

    @staticmethod
    def _create_soil_instance(row):
        """
        Create an instance of the specified soil type.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        BaseSoilType
            Instance of the specified soil type.

        Raises
        ------
        AttributeError
            If the specified class is not found in the `site` module.
        """
        try:
            soil_type = str(row["soil_type"])
            soil_module = importlib.import_module(".site", package=__package__)
            SoilClass = getattr(soil_module, soil_type)
        except (AttributeError, ModuleNotFoundError) as e:
            raise AttributeError(
                f"Class {row['soil_type']} was not found in module 'site'. Error: {e}"
            )

        row["damping"] = row.get("damping", DAMPING_DEFAULT)
        earth_pres_coeff = row.get("earth_pres_coeff", EARTH_PRES_COEFF_DEFAULT)
        row["stress_mean"] = row["shear_stress"] * (1 + 2 * earth_pres_coeff) / 3

        soil_class_params = SoilClass.__init__.__code__.co_varnames
        filtered_row = {k: v for k, v in row.items() if k in soil_class_params}

        return SoilClass(**filtered_row)

    @staticmethod
    def _create_soil_from_file(row):
        """
        Create a soil instance from a file.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        SoilType
            Instance of the soil type created from the file.

        Raises
        ------
        ImportError
            If the 'site' module cannot be imported.
        """
        try:
            return site.SoilType.from_published(
                unit_wt=row["unit_wt"], model="sample", fpath=row["deg_curves"]
            )
        except ImportError as e:
            raise ImportError(f"Could not import the 'site' module. Error: {e}")

    @staticmethod
    def _create_soil_from_published(row):
        """
        Create a soil instance from published data.

        Parameters
        ----------
        row : dict
            Dictionary containing soil property data.

        Returns
        -------
        SoilType
            Instance of the soil type created from published data.

        Raises
        ------
        ImportError
            If the 'site' module cannot be imported.
        """
        try:
            return site.SoilType.from_published(
                unit_wt=row["unit_wt"], model=row["author_curves"]
            )
        except ImportError as e:
            raise ImportError(f"Could not import the 'site' module. Error: {e}")


class MotionFactory:
    """
    Factory to create instances of seismic motions from input data and conversion factors.

    Parameters
    ----------
    scale_factor : float, optional
        Conversion factor to scale the motions. Default is 1.0.
    """

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor

    def create_motions(self, motion_id, motion_df, sample_interval):
        """
        Generates seismic motions for each available component in the input DataFrame.

        Parameters
        ----------
        motion_id : str
            Unique identifier for the seismic motion.
        motion_df : pd.DataFrame
            DataFrame containing acceleration data for each component.
        sample_interval : float
            Sampling interval in seconds.

        Returns
        -------
        dict
            Dictionary where keys are component names and values are instances of TimeSeriesMotion.
        """
        return {
            component: motion.TimeSeriesMotion(
                motion_id,
                f"{motion_id}_{component}",
                sample_interval,
                motion_df[component].values * self.scale_factor,
            )
            for component in MOTION_COMPONENTS
            if component in motion_df.columns
        }


class SeismicResponse:
    """
    Class to analyze the seismic response of site profiles.

    This class constructs site profiles, calculates the seismic response for
    each combination of profile and seismic motion, and stores the results.

    Parameters
    ----------
    motion_dict : dict, optional
        Dictionary containing seismic motion data with their identifiers.
    profile_dict : dict, optional
        Dictionary containing site profiles to analyze with their identifiers.
    scale_factor : float, optional
        Conversion factor to scale the motions. Default is 1.0.
    """

    def __init__(self, motion_dict=None, profile_dict=None, scale_factor=1.0):
        logger.info("Initializing Seismic Response Analyzer")
        self.motion_dict = motion_dict or {}
        self.profile_dict = profile_dict or {}
        self.result_dict = defaultdict(lambda: defaultdict(dict))
        self.motion_factory = MotionFactory(scale_factor)

    def build_profile(self, df):
        """
        Constructs and automatically discretizes a site profile based on an input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing site profile data, including properties of each layer.

        Returns
        -------
        Profile
            Discretized site profile ready for seismic analysis.
        """
        logger.info("Building profile from DataFrame")
        layers = [
            site.Layer(SoilFactory.get_soil(row), row["thickness"], row["shear_vel"])
            for _, row in df.iterrows()
            if SoilFactory.get_soil(row) is not None
        ]
        return site.Profile(layers).auto_discretize()

    def calculate_response(self, profile, motion, type_motion):
        """
        Calculates the seismic response for a specific profile and motion.

        Parameters
        ----------
        profile : Profile
            Site profile to analyze.
        motion : TimeSeriesMotion
            Seismic motion to apply to the profile.
        type_motion : str
            Type of seismic motion (e.g., 'surface' or 'bedrock').

        Returns
        -------
        OutputCollection
            Collection of outputs including response spectra and acceleration time series.
        """
        calc = propagation.EquivalentLinearCalculator()
        calc(motion, profile, profile.location(type_motion, index=-1))

        outputs = output.OutputCollection(
            [
                output.AccelerationTSOutput(
                    location=output.OutputLocation(type_motion, index=0)
                ),
                output.FourierAmplitudeSpectrumOutput(
                    freq_range, output.OutputLocation(type_motion, index=0)
                ),
                output.ResponseSpectrumOutput(
                    freq_range,
                    output.OutputLocation(type_motion, index=0),
                    DAMPING_DEFAULT,
                ),
            ]
        )
        outputs(calc)
        return outputs

    def process_scenarios(self):
        """
        Processes all possible scenarios by combining each site profile with each available seismic motion.

        For each combination of profile and motion, calculates and stores the seismic response.
        """
        logger.info("Processing scenarios")

        def process_profile_motion(motion_id, motion_data, profile_id, profile_df):
            motion_df, type_motion, sample_interval = (
                motion_data["motion"],
                motion_data["type_motion"],
                motion_data["sample_interval"],
            )
            motions = self.motion_factory.create_motions(motion_id, motion_df, sample_interval)
            profile = self.build_profile(profile_df)
            logger.info("Processing profile %s and motion %s", profile_id, motion_id)
            self.result_dict[profile_id][motion_id] = self.store_outputs(motions, profile, type_motion)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_profile_motion, motion_id, motion_data, profile_id, profile_df)
                for motion_id, motion_data in self.motion_dict.items()
                for profile_id, profile_df in self.profile_dict.items()
            ]
            for future in futures:
                future.result()

    def store_outputs(self, motions, profile, type_motion):
        """
        Executes the seismic response calculation and stores the results for each motion component.

        Parameters
        ----------
        motions : dict
            Dictionary of seismic motions generated by the motion factory.
        profile : Profile
            Site profile used in the calculation.
        type_motion : str
            Type of seismic motion ('outcrop' for surface or 'within' for bedrock).

        Returns
        -------
        dict
            Dictionary with output results for each motion component.
        """
        component_results = {}
        for component, motion in motions.items():
            outputs = self.calculate_response(profile, motion, type_motion)
            component_results[component] = {
                output_type: outputs[OUTPUT_TYPES_SITE_RESPONSE[output_type]["index"]]
                .to_dataframe()
                .reset_index()
                for output_type in OUTPUT_TYPES_SITE_RESPONSE.keys()
            }
        return component_results

    def get_results(self):
        """
        Gets the results of the processed seismic analyses.

        Returns
        -------
        defaultdict
            Nested dictionary of results where each combination of profile and
            motion has its responses stored by component.
        """
        logger.info("Getting results")
        return self.result_dict
