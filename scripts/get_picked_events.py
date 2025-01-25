import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import pandas as pd
from obspy import read_inventory

from modules.reader.seed_reader import read_seeds
from modules.processor.preprocess import preprocess_streams
from libs.utils.tools import log_execution_time


def get_sample_rate(inventory, network: str, station: str, location: str) -> float:
    """
    Get the sample rate for a specific sensor.

    Parameters
    ----------
    inventory : Inventory
        Inventory object containing sensor information.
    network : str
        Network code.
    station : str
        Station code.
    location : str
        Location code.

    Returns
    -------
    float
        Sample rate of the sensor.
    """
    selected = inventory.select(network=network, station=station, location=location)
    sample_rate = next(
        (station[0].sample_rate for network in selected for station in network),
        None,
    )
    return sample_rate


def get_sensor_model(inventory, network: str, station: str, location: str) -> str:
    """
    Get the sensor model for a specific sensor.

    Parameters
    ----------
    inventory : Inventory
        Inventory object containing sensor information.
    network : str
        Network code.
    station : str
        Station code.
    location : str
        Location code.

    Returns
    -------
    str
        Model code of the sensor.
    """
    selected = inventory.select(network=network, station=station, location=location)
    model = next(
        (
            prefixes.pop()
            for network in selected
            for station in network
            if len(prefixes := {channel.code[:2] for channel in station}) == 1
        ),
        None,
    )
    return model

@log_execution_time(module="picked_events")
def exec_get_picked_events(
    network: str,
    station: str,
    location: str,
    detrend_types: list,
    conversion_factor: float,
    data_path: str,
    earthquake_df: pd.DataFrame,
) -> dict:
    """
    Execute the process of getting picked events and preprocessing the streams.

    Parameters
    ----------
    network : str
        Network code.
    station : str
        Station code.
    location : str
        Location code.
    detrend_types : list
        List of detrend types to apply during preprocessing.
    conversion_factor : float
        Factor to convert the data units.
    data_path : str
        Base path to the data directory.
    earthquake_df : pd.DataFrame
        DataFrame containing earthquake event information.

    Returns
    -------
    dict
        Dictionary containing date strings as keys and preprocessed Stream objects as values.

    Process
    -------
    1. Construct the sensor code and read the inventory file.
    2. Extract the model and sample rate information from the inventory.
    3. Read the SEED data for the events using the `read_seeds` function.
    4. Preprocess the read streams using the `preprocess_streams` function.
    5. Return the dictionary of preprocessed streams.
    """
    
    # 1. Construct the sensor code and read the inventory file.
    sensor_code = f"{network}.{station}.{location}"
    inventory_path = f"./data/processor/inventory/{sensor_code}.xml"
    inventory = read_inventory(inventory_path)

    # 2. Extract the model and sample rate information from the inventory.
    model = get_sensor_model(inventory, network, station, location)
    sample_rate = get_sample_rate(inventory, network, station, location)

    # 3. Read the SEED data for the events using the `read_seeds` function.
    streams = read_seeds(
        earthquake_df=earthquake_df,
        data_path=data_path,
        network=network,
        station=station,
        location=location,
        model=model,
    )

    # 4. Preprocess the read streams using the `preprocess_streams` function.
    corrected_streams = preprocess_streams(
        streams=streams,
        inventory=inventory,
        detrend_types=detrend_types,
        conversion_factor=conversion_factor,
    )

    # 5. Return the dictionary of preprocessed streams.
    return corrected_streams


if __name__ == "__main__":

    # Main variables
    network = "AT"
    station = "CHABU"
    location = "00"
    detrend_types = ["linear", "demean"]
    conversion_factor = 100  # m/s^2 to cm/s^2
    data_path = "./data/processor/events/"

    # Earthquake info
    earthquake_data = {"date": ["28/12/2024 18:21:00"], "duration": [200]}

    # Conversion to DataFrame
    earthquake_df = pd.DataFrame(earthquake_data)

    corrected_streams = exec_get_picked_events(
        network=network,
        station=station,
        location=location,
        detrend_types=detrend_types,
        conversion_factor=conversion_factor,
        data_path=data_path,
        earthquake_df=earthquake_df,
    )
