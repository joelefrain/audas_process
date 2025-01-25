from obspy.core.stream import Stream
from obspy.core.inventory import Inventory
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def deattach_response(stream: Stream, inventory: Inventory) -> Stream:
    """
    Correct the input stream using the provided inventory.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream to be corrected.
    inventory : obspy.core.inventory.inventory.Inventory
        Inventory containing the response information.

    Returns
    -------
    obspy.core.stream.Stream
        Corrected stream.
    """
    stream.attach_response(inventory)

    # Remove the instrument response from the stream
    stream.remove_sensitivity()

    return stream


def detrend_stream(stream: Stream, type: str = "linear") -> Stream:
    """
    Detrend the input stream.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream to be detrended.
    type : str
        Type of detrending to apply. Options are 'linear', 'constant', 'demean', 'simple'.

    Returns
    -------
    obspy.core.stream.Stream
        Detrended stream.
    """
    stream.detrend(type=type)
    return stream


def convert_units(stream: Stream, factor: float) -> Stream:
    """
    Multiply the data of the input stream by a conversion factor.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream whose data will be multiplied.
    factor : float
        Conversion factor to multiply the stream data by.

    Returns
    -------
    obspy.core.stream.Stream
        Stream with multiplied data.
    """
    for trace in stream:
        trace.data *= factor
    return stream


def stream_to_numpy(stream: Stream) -> np.ndarray:
    """
    Convert the input stream to a numpy array.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream to be converted.

    Returns
    -------
    np.ndarray
        Numpy array containing the stream data.
    """
    return np.array([trace.data for trace in stream])


def correct_by_instrument(
    stream: Stream,
    inventory: Inventory,
    detrend_types: list,
    conversion_factor: float = 100,
) -> Stream:
    """
    Process the input stream by attaching response, removing sensitivity, detrending, and converting units.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        Input stream to be processed.
    inventory : obspy.core.inventory.inventory.Inventory
        Inventory containing the response information.
    detrend_types : list of str
        List of detrending types to apply sequentially. Options are 'linear', 'constant', 'demean', 'simple'.
    conversion_factor : float, optional
        Conversion factor to multiply the stream data by. Default is 100.

    Returns
    -------
    obspy.core.stream.Stream
        Processed stream.
    """
    # Attach response and remove sensitivity
    stream = deattach_response(stream, inventory)

    # Apply each detrend type in the provided order
    for detrend_type in detrend_types:
        stream = detrend_stream(stream, type=detrend_type)

    # Convert units
    stream = convert_units(stream, conversion_factor)

    return stream


def preprocess_streams(
    streams: dict, inventory: Inventory, detrend_types: list, conversion_factor: int = 100
) -> dict:
    """
    Preprocess multiple streams concurrently.

    Parameters
    ----------
    streams : dict
        Dictionary containing date strings as keys and Stream objects as values.
    inventory : obspy.core.inventory.inventory.Inventory
        Inventory containing the response information.
    detrend_types : list of str
        List of detrending types to apply sequentially. Options are 'linear', 'constant', 'demean', 'simple'.
    conversion_factor : int, optional
        Conversion factor to multiply the stream data by. Default is 100.
        
    Returns
    -------
    dict
        Dictionary containing date strings as keys and processed Stream objects as values.
    """
    corrected_streams = {}

    def preprocess_stream(index_str: str, stream: Stream) -> tuple:

        # Correct the stream using the `correct_by_instrument` function
        corrected_stream = correct_by_instrument(
            stream=stream,
            inventory=inventory,
            detrend_types=detrend_types,
            conversion_factor=conversion_factor,
        )

        return index_str, corrected_stream

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda item: preprocess_stream(*item), streams.items())

    for index_str, corrected_stream in results:
        corrected_streams[index_str] = corrected_stream

    return corrected_streams
