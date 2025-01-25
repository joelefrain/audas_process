import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import json
import numpy as np
import pandas as pd

from modules.site_response.factory import SeismicResponse
from modules.displacement_calculator.semiempiric import DisplacementCalculator
from libs.utils.tools import (
    set_name,
    make_dir,
    log_execution_time,
    MotionReader,
    ProfileReader,
)
from libs.utils.logger_config import get_logger
from libs.utils.config_variables import (
    OUTPUT_TYPES_SITE_RESPONSE,
    TABLE_SEP_FORMAT,
    METADATA_CODE,
)

SPC_CODE = OUTPUT_TYPES_SITE_RESPONSE["ResponseSpectrumOutput"]["code"]

module = "get_displacement"
logger = get_logger(module)


@log_execution_time(module="site_response")
def read_data_site_response(
    motion_dir: str, profile_dir: str, sensors_to_use: list = None
) -> tuple[dict, dict]:
    """
    Read motion and profile data for site response analysis.
    """
    logger.info("Reading motion data.")
    print("Hola")
    motion_reader = MotionReader(motion_dir, sensors_to_use)
    motion_dict = motion_reader.read_motions()
    print("Mundo")

    logger.info("Reading profile data.")
    profile_reader = ProfileReader(profile_dir)
    profile_dict = profile_reader.read_profiles()

    return motion_dict, profile_dict


@log_execution_time(module="site_response")
def run_site_response(motion_dict: dict, profile_dict: dict) -> dict:
    """
    Run the site response analysis and save the results.
    """
    logger.info("Creating instance of seismic response analyzer.")
    analyzer = SeismicResponse(motion_dict, profile_dict)

    logger.info("Processing motion and profile scenarios.")
    analyzer.process_scenarios()

    logger.info("Getting and saving results.")
    results = analyzer.get_results()

    return results


@log_execution_time(module="site_response")
def save_data_site_response(
    amplified_seismo_data: dict, output_dir: str, structure: str
) -> None:
    """
    Save the site response analysis results to CSV files.
    """
    logger.info("Saving results to CSV.")
    for profile_id, motions in amplified_seismo_data.items():
        for motion_id, components in motions.items():
            for component_id, outputs in components.items():
                for output_type, output_metadata in OUTPUT_TYPES_SITE_RESPONSE.items():
                    if output_metadata is None:
                        continue
                    output_df = outputs.get(output_type)
                    if output_df is not None:
                        fname = set_name(
                            hierarchy_objects=[
                                structure,
                                motion_id,
                                component_id,
                                profile_id,
                            ],
                            type_objects=[output_metadata["code"]],
                        )
                        output_path = os.path.join(output_dir, f"{fname}.csv")
                        logger.info(f"Saving CSV result to {output_path}")
                        output_df.to_csv(output_path, sep=TABLE_SEP_FORMAT)


@log_execution_time(module="displacement_calculator")
def get_spc_acc(amplified_seismo_data: dict) -> pd.DataFrame:
    spc_acc_data = []

    for profile_id, motions in amplified_seismo_data.items():
        for motion_id, components in motions.items():
            for component_id, outputs in components.items():
                output_df = outputs.get("ResponseSpectrumOutput")

                # Extract frequencies and spectral accelerations
                freqs = output_df.iloc[:, 0]
                spc_acc = output_df.iloc[:, 1]

                # Convert frequencies to periods (Period = 1 / Frequency)
                periods = 1 / freqs

                # Create a temporary DataFrame with periods and spectral accelerations
                temp_df = pd.DataFrame({"periods": periods, "spc_acc": spc_acc})

                # Append the temporary DataFrame to the result list
                spc_acc_data.append(temp_df)

    # Concatenate all temporary DataFrames into one combined DataFrame
    combined_df = pd.concat(spc_acc_data, ignore_index=True)

    # Group by 'Period' and calculate the mean of 'SPC_Acc'
    avg_spc_acc_df = combined_df.groupby("periods", as_index=False).agg(
        {"spc_acc": "mean"}
    )

    # Sort by 'Period' in ascending order
    avg_spc_acc_df = avg_spc_acc_df.sort_values(by="periods").reset_index(drop=True)

    return avg_spc_acc_df


@log_execution_time(module="displacement_calculator")
def run_displacement_calculator(
    method_displacement_calc: str, params_displacement_calc: dict
) -> dict:
    """
    Run the displacement calculation and return the displacement.
    """
    logger.info("Starting displacement calculation")
    try:
        displacement_calculator = DisplacementCalculator(method_displacement_calc)
        displacement_calculator.execute(**params_displacement_calc)
        displacement_attr = displacement_calculator.get_result()
        return displacement_attr
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return None


@log_execution_time(module="displacement_calculator")
def save_data_displacement_calculator(
    displacement_attr: dict, output_dir: str, structure: str
) -> None:
    def default_serializer(obj):
        if isinstance(obj, (list, np.ndarray)):
            return obj.tolist()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    fname = set_name(
        hierarchy_objects=[structure],
        type_objects=[METADATA_CODE],
    )
    json_output_path = os.path.join(output_dir, f"{fname}.json")
    with open(json_output_path, "w") as json_file:
        json.dump(displacement_attr, json_file, default=default_serializer)
    logger.info(f"Displacement data saved to {json_output_path}")


@log_execution_time(module="get_displacement")
def exec_get_displacement(
    company: str,
    project: str,
    structure: str,
    event: str,
    sensors_to_use: list,
    method_displacement_calc: str,
    params_displacement_calc: dict,
    base_output: str,
    base_data: str,
) -> dict:
    """
    Calculate the seismic displacement induced in a single geotechnical structure using data from multiple sensors for a single seismic event.

    This function performs the following steps:
    1. Reads motion and profile data for site response analysis.
    2. Runs the site response analysis and saves the results.
    3. Calculates the spectral acceleration at the degraded period.
    4. Runs the displacement calculation and saves the results.

    Parameters
    ----------
    company : str
        The name of the company.
    project : str
        The name of the project.
    structure : str
        The name of the geotechnical structure.
    event : str
        The identifier of the seismic event.
    sensors_to_use : list
        List of sensors to use for the analysis.
    method_displacement_calc : str
        The method to use for displacement calculation.
    params_displacement_calc : dict
        Parameters for the displacement calculation method.
    base_output : str
        Base directory for output files.
    base_data : str
        Base directory for input data files.

    Returns
    -------
    dict
        A dictionary containing the displacement attributes, spectral acceleration at the degraded period, and the degraded structural period.
    """

    # Paths
    motion_dir = f"{base_output}/motion_divider/{company}/{project}/{event}"
    profile_dir = f"{base_data}/{company}/{project}/{structure}"
    site_response_dir = make_dir(
        base_dir=f"{base_output}/site_response",
        path_components=[company, project, structure, event],
    )
    displacement_calculator_dir = make_dir(
        base_dir=f"{base_output}/displacement_calculator",
        path_components=[company, project, event],
    )

    # 1. Seismic Response Analysis
    motion_dict, profile_dict = read_data_site_response(
        motion_dir=motion_dir,
        profile_dir=profile_dir,
        sensors_to_use=sensors_to_use,
    )

    amplified_seismo_data = run_site_response(
        motion_dict=motion_dict,
        profile_dict=profile_dict,
    )

    save_data_site_response(
        amplified_seismo_data=amplified_seismo_data,
        output_dir=site_response_dir,
        structure=structure,
    )

    # 2. Displacement Calculation
    avg_spc_acc = get_spc_acc(amplified_seismo_data=amplified_seismo_data)

    displacement_attr = run_displacement_calculator(
        method_displacement_calc=method_displacement_calc,
        params_displacement_calc={
            **params_displacement_calc,
            "spc_acc": avg_spc_acc["spc_acc"].tolist(),
            "periods": avg_spc_acc["periods"].tolist(),
        },
    )

    save_data_displacement_calculator(
        displacement_attr=displacement_attr,
        output_dir=displacement_calculator_dir,
        structure=structure,
    )

    return displacement_attr


if __name__ == "__main__":

    # Main variables
    company = "Shahuindo_SAC"
    project = "Shahuindo"
    structure = "Pad2G"
    event = "2024-EAR-0001"
    sensors_to_use = ["AN.AUDAS.00", "AN.RSAQP."]

    # Variables for displacement calculation
    method_displacement_calc = "Bray2018Calculator"
    params_displacement_calc = {
        "fail_surf_type": "block",
        "h_slide_block": 25,
        "vs_prom_fail_surf": 350,
        "yield_acc": 0.19,
        "magnitude": 8,
        "desv_std": 0,
    }

    # Base directories
    base_output = "./outputs"
    base_data = "./data/site_response"
    motion_path = "motion_divider"

    displacement_attr = exec_get_displacement(
        company=company,
        project=project,
        structure=structure,
        event=event,
        sensors_to_use=sensors_to_use,
        method_displacement_calc=method_displacement_calc,
        params_displacement_calc=params_displacement_calc,
        base_output=base_output,
        base_data=base_data,
    )
