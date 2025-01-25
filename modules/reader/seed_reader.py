import os
import sys

# Add 'libs' path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from obspy import read, UTCDateTime
from datetime import datetime, timedelta
from config.config_variables import CHANNEL_INFO
from config.logger_config import get_logger

module = "historical_monitoring"
logger = get_logger(module)


def read_earthquakes(earthquake_df, sensor_info, base_path):
    """
    Reads earthquake data for each row in the DataFrame and returns a dictionary of streams.

    Parameters
    ----------
    earthquake_df : pd.DataFrame
        DataFrame containing earthquake data.
    sensor_info : dict
        Dictionary containing sensor information.
    base_path : str
        Base path to the data files.

    Returns
    -------
    dict
        Dictionary with date strings as keys and corresponding streams as values.
    """
    streams = {}
    logger.info("Starting to read earthquake data")

    for index, row in earthquake_df.iterrows():
        date_str = row["date"]
        dt = parse_date(date_str)
        year = dt.year
        day_number = dt.strftime("%j")
        start_time = UTCDateTime(dt)
        end_time = start_time + timedelta(seconds=row["duration"])

        network, station, location = (
            sensor_info["network"],
            sensor_info["station"],
            sensor_info["location"],
        )
        logger.info(
            f"Parsing earthquake data for {date_str} - {network}.{station}.{location}"
        )

        st = read_stream_for_event(
            sensor_info, base_path, year, day_number, start_time, end_time
        )
        if st is None:
            logger.warning(f"No valid stream found for date: {date_str}")
        else:
            logger.info(f"Stream successfully read for date: {date_str}")

        streams[date_str] = st

    logger.info("Finished reading earthquake data")
    return streams


def parse_date(date_str):
    """
    Parses a date string into a datetime object.

    Parameters
    ----------
    date_str : str
        Date string in the format "%d/%m/%Y %H:%M:%S".

    Returns
    -------
    datetime
        Parsed datetime object.
    """
    logger.debug(f"Parsing date string: {date_str}")
    return datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")


def build_file_path(base_path, sensor_info, year, day_number, component):
    """
    Constructs the file path for a specific component of the sensor.

    Parameters
    ----------
    base_path : str
        Base path to the data files.
    sensor_info : dict
        Dictionary containing sensor information.
    year : int
        Year of the event.
    day_number : str
        Day number of the year.
    component : str
        Component of the sensor.

    Returns
    -------
    str
        Constructed file path.
    """
    path = (
        f"{base_path}/{year}/{sensor_info['network']}/{sensor_info['station']}/"
        f"{sensor_info['model']}{component}.D/"
        f"{sensor_info['network']}.{sensor_info['station']}.{sensor_info['location']}."
        f"{sensor_info['model']}{component}.D.{year}.{day_number}"
    )
    logger.debug(f"Constructed file path: {path}")
    return path


def read_stream_for_event(
    sensor_info, base_path, year, day_number, start_time, end_time
):
    """
    Reads and filters the stream data for a specific event based on time range.
    Handles events that span multiple days.

    Parameters
    ----------
    sensor_info : dict
        Dictionary containing sensor information.
    base_path : str
        Base path to the data files.
    year : int
        Year of the event.
    day_number : str
        Day number of the year.
    start_time : UTCDateTime
        Start time of the event.
    end_time : UTCDateTime
        End time of the event.

    Returns
    -------
    Stream
        Stream object containing the filtered data.
    """
    st = None  # Initialize an empty Stream object
    logger.debug(f"Reading stream for event: year={year}, day_number={day_number}")

    # Iterate over components and read streams for the start day
    for component in CHANNEL_INFO.keys():
        st = read_and_append_stream(
            base_path,
            sensor_info,
            year,
            day_number,
            component,
            start_time,
            end_time,
            st,
        )

    # If the event spans multiple days, process the next day
    if start_time.date != end_time.date:
        next_day = start_time + timedelta(days=1)
        next_year = next_day.year
        next_day_number = next_day.strftime("%j")

        for component in CHANNEL_INFO.keys():
            st = read_and_append_stream(
                base_path,
                sensor_info,
                next_year,
                next_day_number,
                component,
                start_time,
                end_time,
                st,
            )

    return st


def read_and_append_stream(
    base_path, sensor_info, year, day_number, component, start_time, end_time, st
):
    """
    Reads a stream for a specific component and appends valid traces to the main stream.

    Parameters
    ----------
    base_path : str
        Base path to the data files.
    sensor_info : dict
        Dictionary containing sensor information.
    year : int
        Year of the event.
    day_number : str
        Day number of the year.
    component : str
        Component of the sensor.
    start_time : UTCDateTime
        Start time of the event.
    end_time : UTCDateTime
        End time of the event.
    st : Stream
        Main Stream object to append traces to.

    Returns
    -------
    Stream
        Updated Stream object with appended traces.
    """
    path = build_file_path(base_path, sensor_info, year, day_number, component)
    if os.path.exists(path):
        logger.debug(f"File exists: {path}")
        stream = read(path, starttime=start_time, endtime=end_time)
        st = append_valid_traces(st, stream, start_time, end_time)
    else:
        logger.warning(f"File does not exist: {path}")
    return st


def append_valid_traces(st, stream, start_time, end_time):
    """
    Appends valid traces from the stream to the main Stream object.

    Parameters
    ----------
    st : Stream
        Main Stream object to append traces to.
    stream : Stream
        Stream object containing traces to be appended.
    start_time : UTCDateTime
        Start time of the event.
    end_time : UTCDateTime
        End time of the event.

    Returns
    -------
    Stream
        Updated Stream object with appended traces.
    """
    for trace in stream:
        if trace.stats.starttime <= end_time and trace.stats.endtime >= start_time:
            if st is None:
                st = stream.__class__()  # Initialize the Stream if not already
            st.append(trace)
            logger.debug(f"Appended trace: {trace.id}")
    return st
