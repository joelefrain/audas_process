from obspy import read, read_inventory
from obspy.core.stream import Stream
from obspy.core.inventory import Inventory
import numpy as np

def correct_stream(stream, inventory):
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

def detrend_stream(stream, type='linear'):
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

def stream_to_numpy(stream):
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

# Example usage
if __name__ == "__main__":
    # Load a sample stream and inventory
    stream = read("./data/test_processor/AN.AUDAS.01.HHE.D.2024.197")
    inventory = read_inventory("./data/test_processor/anddes_anddes_new.xml")
    # inventory = read_inventory("./data/test_processor/anddes_anta.xml")
    
    print(stream)
    print(inventory)
    
    # Correct the stream
    corrected_stream = correct_stream(stream, inventory)
    
    # # Detrend the corrected stream
    detrended_stream = detrend_stream(corrected_stream)
    
    # # Convert the detrended stream to numpy array
    numpy_array = stream_to_numpy(detrended_stream)
    
    # # Print the numpy array
    print(numpy_array)    
    
    # # Verifica los IDs del Stream
    # for trace in stream:
    #     print(trace.id)

    
    
    # # Verifica los IDs del Inventory
    # for network in inventory:
    #     for station in network:
    #         for channel in station:
    #             print(channel.code)
    
    


    
