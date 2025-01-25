import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime
from obspy import read, Stream
from obspy.core.utcdatetime import UTCDateTime
from libs.utils.config_variables import CHANNEL_INFO
from libs.utils.logger_config import get_logger

module = "reader"
logger = get_logger(module)


def get_stream_path(
    data_path: str,
    year: int,
    network: str,
    station: str,
    model: str,
    component: str,
    location: str,
    day_number: str,
) -> str:
    """
    Construct the file path for a given component.

    Parameters
    ----------
    data_path : str
        Base path to the data directory.
    year : int
        Year of the event.
    network : str
        Network code.
    station : str
        Station code.
    model : str
        Model code.
    component : str
        Component code.
    location : str
        Location code.
    day_number : str
        Day of the year in Julian format.

    Returns
    -------
    str
        Constructed file path.
    """
    return (
        f"{data_path}/{year}/{network}/{station}/{model}{component}.D/"
        f"{network}.{station}.{location}.{model}{component}.D.{year}.{day_number}"
    )


def read_component(path: str, start_time: UTCDateTime, end_time: UTCDateTime) -> Stream:
    """
    Read a component from the given path within the specified time range.

    Parameters
    ----------
    path : str
        Path to the data file.
    start_time : UTCDateTime
        Start time for reading the data.
    end_time : UTCDateTime
        End time for reading the data.

    Returns
    -------
    Stream
        Stream object containing the read data.
    """
    if os.path.exists(path):
        stream = read(path, starttime=start_time, endtime=end_time)
        for trace in stream:
            if trace.stats.starttime <= end_time and trace.stats.endtime >= start_time:
                logger.info(f"Read trace from {path}")
                return trace
    logger.warning(f"Path does not exist or no valid trace found: {path}")
    return None


def read_event(
    row: dict, data_path: str, network: str, station: str, location: str, model: str
) -> tuple:
    """
    Process a single event and read its components.

    Parameters
    ----------
    row : dict
        Dictionary containing event information.
    data_path : str
        Base path to the data directory.
    network : str
        Network code.
    station : str
        Station code.
    location : str
        Location code.
    model : str
        Model code.

    Returns
    -------
    tuple
        Tuple containing the date string and the Stream object.
    """
    date_str = row["date"]
    dt = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
    year = dt.year
    day_number = dt.strftime("%j")
    start_time = UTCDateTime(dt)
    end_time = start_time + row["duration"]

    st = Stream()  # Initialize an empty Stream object

    for component in CHANNEL_INFO.keys():
        path = get_stream_path(
            data_path, year, network, station, model, component, location, day_number
        )
        trace = read_component(path, start_time, end_time)
        if trace:
            st.append(trace)

    if len(st) > 0:
        logger.info(f"Successfully read event for date: {date_str}")
    else:
        logger.warning(f"No valid streams found for event on date: {date_str}")

    return date_str, st if len(st) > 0 else None


def read_seeds(
    earthquake_df, data_path: str, network: str, station: str, location: str, model: str
) -> dict:
    """
    Read SEED data for multiple events.

    Parameters
    ----------
    earthquake_df : DataFrame
        DataFrame containing earthquake event information.
    data_path : str
        Base path to the data directory.
    network : str
        Network code.
    station : str
        Station code.
    location : str
        Location code.
    model : str
        Model code.

    Returns
    -------
    dict
        Dictionary containing date strings as keys and Stream objects as values.
    """
    streams = {}

    def read_seed(row):
        date_str, st = read_event(row, data_path, network, station, location, model)
        return date_str, st

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(read_seed, row) for index, row in earthquake_df.iterrows()
        ]
        for future in as_completed(futures):
            date_str, st = future.result()
            if st:
                streams[date_str] = st
                logger.info(f"Added stream for date: {date_str}")
            else:
                logger.warning(f"Stream for date {date_str} is empty or not found")

    return streams
