# spectral_accel.pyx
from libc.math cimport sqrt, sin, cos, exp, atan2, pi
import numpy as np
cimport numpy as np

def calculate_spectral_accel(np.ndarray[np.float64_t, ndim=1] data, double delta, np.ndarray[np.float64_t, ndim=1] T, double xi):
    cdef int data_len = data.shape[0]
    cdef int T_len = T.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(T_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] c = np.zeros(T_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] wd = np.zeros(T_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] p1 = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] S = np.zeros(T_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] u1 = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] udre1 = np.zeros(data_len, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] udd1 = np.zeros(data_len, dtype=np.float64)

    cdef double dt = 1.0 / delta
    cdef double mass = 1.0
    cdef double I0, J0, AA_00, AA_01, AA_10, AA_11, BB_00, BB_01, BB_10, BB_11
    cdef double wj, wdj
    cdef int j, xx

    for j in range(T_len):
        w[j] = 2 * pi / T[j]
        c[j] = 2 * xi * w[j] * mass
        wd[j] = w[j] * sqrt(1 - xi * xi)

    p1[:] = -mass * data

    for j in range(T_len):
        wj = w[j]
        wdj = wd[j]
        I0 = 1.0 / (wj * wj) * (1 - exp(-xi * wj * dt) * (xi * wj / wdj * sin(wdj * dt) + cos(wdj * dt)))
        J0 = 1.0 / (wj * wj) * (xi * wj + exp(-xi * wj * dt) * (-xi * wj * cos(wdj * dt) + wdj * sin(wdj * dt)))

        AA_00 = exp(-xi * wj * dt) * (cos(wdj * dt) + xi * wj / wdj * sin(wdj * dt))
        AA_01 = exp(-xi * wj * dt) * sin(wdj * dt) / wdj
        AA_10 = -wj * wj * exp(-xi * wj * dt) * sin(wdj * dt) / wdj
        AA_11 = exp(-xi * wj * dt) * (cos(wdj * dt) - xi * wj / wdj * sin(wdj * dt))

        BB_00 = I0 * (1 + xi / (wj * dt)) + J0 / (wj * wj * dt) - 1.0 / (wj * wj)
        BB_01 = -xi / (wj * dt) * I0 - J0 / (wj * wj * dt) + 1.0 / (wj * wj)
        BB_10 = J0 - (xi * wj + 1.0 / dt) * I0
        BB_11 = I0 / dt

        u1[0] = 0.0
        udre1[0] = 0.0

        for xx in range(1, data_len):
            u1[xx] = AA_00 * u1[xx - 1] + AA_01 * udre1[xx - 1] + BB_00 * p1[xx - 1] + BB_01 * p1[xx]
            udre1[xx] = AA_10 * u1[xx - 1] + AA_11 * udre1[xx - 1] + BB_10 * p1[xx - 1] + BB_11 * p1[xx]

        udd1[:] = -(wj * wj * u1 + c[j] * udre1) - data
        S[j] = np.max(np.abs(udd1 + data))

    return S
