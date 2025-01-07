# Default type of motion for site response analysis
TYPE_MOTION_DEFAULT = "outcrop"

# Frequency range parameters for logspace
LOG_FREQ_PARAMS = [-2, 2, 500]

# Default damping ratio
DAMPING_DEFAULT = 0.05

# Default earth pressure coefficient
EARTH_PRES_COEFF_DEFAULT = 1

# Components of motion
MOTION_COMPONENTS = ["EW", "NS", "UD"]
MOTION_COLORS = ["b", "g", "r"]

# Output types for site response analysis
OUTPUT_TYPES_SITE_RESPONSE = {
    "AccelerationTSOutput": {"index": 0, "code": "ACC"},
    "FourierAmplitudeSpectrumOutput": {"index": 1, "code": "FOU"},
    "ResponseSpectrumOutput": {"index": 2, "code": "SPC"},
}

# Separator format for CSV files
TABLE_SEP_FORMAT = ";"

# Union format for filenames
UNION_FORMAT_BTW_OBJECTS = "_"
UNION_FORMAT_FOR_HIERARCHY = "."

# Global coefficients for Bray et al. (2018) method
BRAY_ZERO_DISPL_PROB_COEFFS = [-2.64, -3.2, -0.17, -0.49, 2.09, 2.91]
BRAY_DISP_AVG_COEFFS = [-6.896, 3.081, -0.803, -3.353, -0.39, 0.538, 3.06, -0.225, 0.55]
COEFF_STRUCT_PERIOD = {"block": 2.6, "circular": 4.0}
