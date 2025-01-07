import os

from modules.site_response.factory import SeismicResponse
from modules.displacement_calculator.semiempiric import DisplacementCalculator
from libs.utils.tools import (
    set_name,
    make_dir,
    read_json,
    log_execution_time,
    MotionReader,
    ProfileReader,
)
from libs.utils.logger_config import get_logger
from libs.utils.config_variables import (
    OUTPUT_TYPES_SITE_RESPONSE,
    TABLE_SEP_FORMAT,
)

module = "main"
logger = get_logger(module)


@log_execution_time(module="site_response")
def read_data_site_response(motion_dir, profile_dir):
    """
    Read motion and profile data for site response analysis.
    """
    logger.info("Reading motion data.")
    motion_reader = MotionReader(motion_dir)
    motion_dict = motion_reader.read_motions()

    logger.info("Reading profile data.")
    profile_reader = ProfileReader(profile_dir)
    profile_dict = profile_reader.read_profiles()

    return motion_dict, profile_dict


@log_execution_time(module="site_response")
def save_data_site_response(results, output_dir, structure, module, trigger):
    """
    Save the site response analysis results to CSV files.
    """
    logger.info("Saving results to CSV.")
    for profile_id, motions in results.items():
        for motion_id, components in motions.items():
            for component_id, outputs in components.items():
                for output_type, output_metadata in OUTPUT_TYPES_SITE_RESPONSE.items():
                    if output_metadata is None:
                        continue
                    output_df = outputs.get(output_type)
                    if output_df is not None:
                        fname = set_name(
                            hierarchy_objects=[structure, motion_id, component_id, profile_id],
                            type_objects=[module, trigger, output_metadata["code"]],
                        )
                        output_path = os.path.join(output_dir, f"{fname}.csv")
                        logger.info(f"Saving CSV result to {output_path}")
                        output_df.to_csv(output_path, sep=TABLE_SEP_FORMAT)


@log_execution_time(module="site_response")
def run_site_response(module, trigger, output_dir, structure, motion_dict, profile_dict, scale_factor):
    """
    Run the site response analysis and save the results.
    """
    logger.info("Creating instance of seismic response analyzer.")
    analyzer = SeismicResponse(motion_dict, profile_dict, scale_factor)

    logger.info("Processing motion and profile scenarios.")
    analyzer.process_scenarios()

    logger.info("Getting and saving results.")
    results = analyzer.get_results()

    save_data_site_response(results, output_dir, structure, module, trigger)
    return results


def extract_base_parameters_disp_calc(params: dict) -> tuple[str, dict]:
    """
    Extract the displacement calculation method and base parameters from JSON data.
    """
    method_displacement_calc = params[0]["method_displacement_calc"]
    base_parameters = params[1]
    return method_displacement_calc, base_parameters


def extract_spectral_response(results: dict, base_parameters: dict) -> list[dict]:
    """
    Extract periods and spectral acceleration (spc_acc) from results and create parameter sets.
    """
    parameters_list = []

    for profile_id, motions in results.items():
        for motion_id, components in motions.items():
            for component_id, outputs in components.items():
                output_df = outputs.get("ResponseSpectrumOutput")
                if output_df is not None:
                    parameters = base_parameters.copy()
                    parameters["periods"] = output_df.iloc[:, 0].tolist()
                    parameters["spc_acc"] = output_df.iloc[:, 1].tolist()
                    parameters_list.append(parameters)
                    break

    if not parameters_list:
        logger.warning("No valid parameters found for displacement calculation.")

    return parameters_list


@log_execution_time(module="displacement_calculator")
def read_data_displacement_calculator(params_path: str, results: dict) -> tuple[str, list[dict]]:
    """
    Read data for displacement calculation from a JSON file and results dictionary.
    """
    params = read_json(params_path)
    method_displacement_calc, base_parameters = extract_base_parameters_disp_calc(params)
    parameters_list = extract_spectral_response(results, base_parameters)

    return method_displacement_calc, parameters_list


@log_execution_time(module="displacement_calculator")
def run_displacement_calculator(method_displacement_calc, parameters_displacement_calc_list):
    """
    Run the displacement calculation and return the average displacement.
    """
    logger.info("Starting displacement calculation")
    all_displacements = []
    try:
        for parameters_displacement_calc in parameters_displacement_calc_list:
            displacement_calculator = DisplacementCalculator(method_displacement_calc)
            displacement_calculator.execute(**parameters_displacement_calc)
            displacements = displacement_calculator.get_result()
            if displacements is not None:
                all_displacements.append(displacements)

        if all_displacements:
            average_displacement = sum(all_displacements) / len(all_displacements)
            logger.info("Calculated average seismic displacement: %f", average_displacement)
            return average_displacement
        else:
            logger.warning("No valid displacement values found.")
            return None
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return None


if __name__ == "__main__":
    # Conditions of analysis
    event = "2024-0001"
    trigger_code = "EAR"  # Earthquake

    # Seismic Response Analysis
    module = "SRES"  # "site_response"

    compania_minera = "Shahuindo_SAC"
    proyecto_minero = "Shahuindo"
    estructura_geotecnica = "Pad2G"

    path_components = [compania_minera, proyecto_minero, estructura_geotecnica, event]
    output_dir = make_dir(base_dir="outputs", path_components=path_components)

    structure_info = {
        "name": estructura_geotecnica,
        "scale_factor": 1.0,
        "motion_dir": f"./data/test_site_response/{estructura_geotecnica}/motion",
        "profile_dir": f"./data/test_site_response/{estructura_geotecnica}/profile",
        "output_dir": output_dir,
    }

    # Read motion and profile data
    motion_dict, profile_dict = read_data_site_response(
        structure_info["motion_dir"], structure_info["profile_dir"]
    )

    results = run_site_response(
        module=module,
        trigger=trigger_code,
        output_dir=output_dir,
        structure=structure_info["name"],
        motion_dict=motion_dict,
        profile_dict=profile_dict,
        scale_factor=structure_info["scale_factor"],
    )

    # Displacement Calculation
    params_path = "./data/test_displacement_calculator/params.json"
    method_displacement_calc, parameters_displacement_calc_list = (
        read_data_displacement_calculator(params_path, results)
    )

    average_displacement = run_displacement_calculator(
        method_displacement_calc, parameters_displacement_calc_list
    )
