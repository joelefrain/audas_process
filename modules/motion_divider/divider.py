import os
import numpy as np

def save_motion_data(type_motion, sample_rate, signals, output_dir, filename):
    """
    Save motion data to a text file with the specified format.

    Parameters
    ----------
    type_motion : str
        Type of motion.
    sample_rate : float
        Sample rate of the signals.
    signals : np.ndarray
        Numpy array containing the signals (EW, NS, UD).
    output_dir : str
        Directory where the file will be saved.
    filename : str
        Name of the output file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w') as file:
        # Write the header
        file.write(f"{type_motion};{sample_rate}\n")
        
        # Write the signal data
        for ew, ns in zip(signals[0], signals[1]):
            file.write(f"{ew};{ns}\n")

# Example usage
if __name__ == "__main__":
    type_motion = "outcrop"
    sample_rate = 0.02
    signals = np.array([
        [0.00000000000302, 0.00000000000134, -0.00000000000037],  # EW
        [0.02000000000000, -0.00000000006283, -0.00000000006211],  # NS
        [0.00000000000302, 0.00000000000134, -0.00000000000037]   # UD (not used)
    ])
    output_dir = "./outputs"
    filename = "motion_data.txt"
    
    save_motion_data(type_motion, sample_rate, signals, output_dir, filename)
