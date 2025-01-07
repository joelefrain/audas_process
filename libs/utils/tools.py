import os
import sys
import time
import json

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd

from libs.utils.config_variables import (
    MOTION_COMPONENTS,
    UNION_FORMAT_FOR_HIERARCHY,
    UNION_FORMAT_BTW_OBJECTS,
    TABLE_SEP_FORMAT,
)
from libs.utils.logger_config import get_logger

def read_json(file_path: str) -> dict:
    """
    Read and parse a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.
    """
    with open(file_path, "r") as file:
        return json.load(file)

def name_orientation(name):
    """
    Generate orientation names for motion components.

    Parameters
    ----------
    name : str
        Base name for the orientation.

    Returns
    -------
    list of str
        List of orientation names for each motion component.
    """
    return [
        f"{name}{UNION_FORMAT_FOR_HIERARCHY}{component}"
        for component in MOTION_COMPONENTS
    ]


def set_name(hierarchy_objects=[], type_objects=[]):
    """
    Generate a name based on hierarchy and type objects.

    Args:
        hierarchy_objects (list): List of hierarchy objects.
        type_objects (list): List of type objects.

    Returns:
        str: Generated filename.
    """
    return UNION_FORMAT_BTW_OBJECTS.join(
        [
            UNION_FORMAT_FOR_HIERARCHY.join(hierarchy_objects),
            UNION_FORMAT_BTW_OBJECTS.join(type_objects),
        ]
    )


def print_structure(data, indent=0):
    """
    Imprime la estructura de un diccionario o lista de forma jerárquica.

    Args:
        data: El diccionario o lista a analizar.
        indent: Nivel de indentación para la impresión (usado internamente en la recursión).
    """
    space = "  " * indent  # Espacios para la indentación
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{space}{key}:")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print(f"{space}[{i}]:")
            print_structure(item, indent + 1)
    else:
        print(f"{space}{data}")


def log_execution_time(func=None, *, module=None):
    """
    Wrapper function to log the execution time of a function.

    Parameters
    ----------
    func : function
        The function to be wrapped.
    module : str, optional
        The module name for the logger.

    Returns
    -------
    function
        The wrapped function with execution time logging.
    """
    if func is None:
        return lambda f: log_execution_time(f, module=module)

    logger = get_logger(module) if module else get_logger("default")

    def wrapper(*args, **kwargs):
        start_time = time.time()
        if args and hasattr(args[0], "__class__"):
            class_name = args[0].__class__.__name__
            logger.info(f"Starting execution of {class_name}.{func.__name__}")
        else:
            logger.info(f"Starting execution of {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if args and hasattr(args[0], "__class__"):
            logger.info(
                f"Finished execution of {class_name}.{func.__name__} in {elapsed_time:.2f} seconds"
            )
        else:
            logger.info(
                f"Finished execution of {func.__name__} in {elapsed_time:.2f} seconds"
            )
        return result

    return wrapper


def make_dir(base_dir, path_components):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    base_dir : str
        Base directory where the structure will be created.
    path_components : list of str
        List of path components to be joined.

    Returns
    -------
    str
        The full path of the created directory.
    """
    path = os.path.join(base_dir, *path_components)
    os.makedirs(path, exist_ok=True)
    return path

class MotionReader:

    def __init__(self, motion_dir):
        """
        Initialize the MotionReader with the directory containing motion files.

        Parameters
        ----------
        motion_dir : str
            Directory path where motion files are stored.
        """
        self.motion_dir = motion_dir

    def file_to_df(self, motion_path):
        """
        Read a motion file and convert it into a DataFrame.

        Parameters
        ----------
        motion_path : str
            Path to the motion file.

        Returns
        -------
        tuple
            A tuple containing the type of motion, sample interval, and the DataFrame of motion data.
        """
        type_motion, sample_interval, data = self._read_motion_file(motion_path)
        columns = self._define_columns(data)
        motion_df = pd.DataFrame(data, columns=columns)
        return type_motion, sample_interval, motion_df

    def _read_motion_file(self, motion_path):
        """
        Read the motion file and extract the type of motion, sample interval, and data.

        Parameters
        ----------
        motion_path : str
            Path to the motion file.

        Returns
        -------
        tuple
            A tuple containing the type of motion, sample interval, and the data as a list of lists.
        """
        with open(motion_path) as fp:
            first_line = fp.readline().strip().split(TABLE_SEP_FORMAT)
            type_motion = first_line[0]
            sample_interval = float(first_line[1])
            data = self._read_data_lines(fp)
        return type_motion, sample_interval, data

    def _read_data_lines(self, fp):
        """
        Read and process the data lines from the motion file.

        Parameters
        ----------
        fp : file object
            File object of the motion file.

        Returns
        -------
        list
            List of lists containing the motion data.
        """
        data = []
        for line in fp:
            if line.strip():
                data.append(list(map(float, line.strip().split(TABLE_SEP_FORMAT))))
        return data

    def _define_columns(self, data):
        """
        Define column names based on the number of values in the first data line.

        Parameters
        ----------
        data : list
            List of lists containing the motion data.

        Returns
        -------
        list
            List of column names.
        """
        first_data_line = data[0]
        return MOTION_COMPONENTS[: len(first_data_line)]

    def read_motions(self):
        """
        Read all motion files in the directory and convert them into a dictionary.

        Returns
        -------
        dict
            Dictionary containing motion data with their identifiers.
        """
        motion_dict = {}
        for motion_file in os.listdir(self.motion_dir):
            if motion_file.endswith(".txt"):
                motion_path = os.path.join(self.motion_dir, motion_file)
                type_motion, sample_interval, motion_df = self.file_to_df(motion_path)
                motion_id = os.path.splitext(motion_file)[0]
                motion_dict[motion_id] = {
                    "type_motion": type_motion,
                    "sample_interval": sample_interval,
                    "motion": motion_df,
                }
        return motion_dict


class ProfileReader:

    def __init__(self, profile_dir):
        self.profile_dir = profile_dir

    def read_profiles(self):
        """Leer todos los archivos de perfil del directorio"""
        profile_dict = {}
        for profile_file in os.listdir(self.profile_dir):
            if profile_file.endswith(".csv"):
                profile_path = os.path.join(self.profile_dir, profile_file)
                profile_df = pd.read_csv(profile_path, sep=TABLE_SEP_FORMAT)
                profile_id = os.path.splitext(profile_file)[0]
                profile_dict[profile_id] = profile_df
        return profile_dict
