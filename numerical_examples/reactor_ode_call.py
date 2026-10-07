#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import csv
import math
import numpy as np
from scipy.integrate import solve_ivp
import DBDDSBB
from DBDDSBB._solver import DecomposedProblemEvaluator

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1


SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# PROBLEM SETTINGS
# ============================================================
# Eight temperature blocks: real 273 K, 298 K, 323 K plus synthetic
# interpolated 280 K, 290 K, 300 K, 310 K, and 320 K.
#
# Shared/master variables:
#   s[0] = g1 = ln(k1)
#   s[1] = g5 = ln(k5)
#
# Private variables in each temperature block:
#   k2f_prime = ln(k2f)
#   k3f_prime = ln(k3f)
#
# k4_prime is fixed to the paper/Singer value for the real temperatures
# and to the interpolated value for the synthetic temperatures.
#
# Full problem dimension:
#   D = 2 shared variables + 8*(2 private variables) = 18.
N_BLOCKS = 8
MASTER_DIM = 2
PRIVATE_DIM = 2
FULL_DIM = MASTER_DIM + N_BLOCKS * PRIVATE_DIM

# Search bounds for the two shared/master variables.
# These correspond physically to k1 in [20, 250] and k5 in [500, 5000].
G1_LB, G1_UB = math.log(20.0), math.log(250.0)
G5_LB, G5_UB = math.log(500.0), math.log(5000.0)

# Search bounds for each block's private variables, in log/prime scale.
K2F_PRIME_LB, K2F_PRIME_UB = 2.303, 7.090
K3F_PRIME_LB, K3F_PRIME_UB = 2.303, 7.090

# Lower-bound formulation: choose "F1" or "F2".
FORMULATION = "F2"

# Private-partition refinement:
# M_nu = ceil(M0 * gamma^floor((nu - 1) / L_M)).
# The solver tree level is nu - 1.
M_PRIVATE_0 = 20
M_PRIVATE_GAMMA = 1.2
M_PRIVATE_LEVEL_INTERVAL = 15
SAMPLE_PER_AXIS = 2

# F2 problem-level parameters.
F2_GAMMA0 = 2.0
F2_RHO = 2.0
F2_OMEGA = 0.001

# ODE solves are expensive, so distribute private-grid evaluations
# across the persistent MPI workers.
PARALLEL_FUNCTION_EVALUATIONS = True

# Solve the same 18D problem once for each time limit (seconds).
TIME_LIMITS = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000]

CSV_OUTPUT = f"ode_eight_temperature_star_18d_{FORMULATION}_time_sweep_results.csv"
SAVE_BEST_FIT_CSVS = True

# Configure F2 parameters. These are used when FORMULATION = "F2".
DBDDSBB._DB_underestimator.configure_formulation2(
    gamma0=F2_GAMMA0,
    rho=F2_RHO,
    omega=F2_OMEGA,
)


# ============================================================
# DATA FOR 273 K, 298 K, 323 K + SYNTHETIC INTERPOLATED TEMPERATURES
# ============================================================

def make_temperature_cases():
    t_data = np.round(np.arange(0.01, 4.46 + 0.01, 0.01), 2)

    I_raw_273 = np.array([
        13.88,22.00,23.17,24.13,25.67,26.34,26.34,25.93,25.47,24.96,
        24.71,24.39,23.91,23.43,22.85,22.25,21.85,21.43,21.03,20.69,
        20.21,19.82,19.25,18.81,18.30,17.82,17.52,17.05,16.57,16.18,
        15.85,15.43,15.08,14.84,14.36,14.14,13.67,13.38,13.03,12.78,
        12.52,12.24,12.07,11.86,11.72,11.45,11.13,10.89,10.59,10.37,
        10.07,9.948,9.848,9.762,9.640,9.466,9.308,9.117,9.144,8.896,
        8.786,8.450,8.217,8.145,7.787,7.760,7.682,7.619,7.423,7.327,
        7.175,7.071,6.923,6.687,6.624,6.541,6.508,6.291,6.118,6.045,
        5.973,5.953,5.784,5.761,5.797,5.707,5.712,5.551,5.410,5.210,
        5.196,5.100,5.076,4.949,4.945,4.913,4.813,4.685,4.579,4.470,
        4.453,4.377,4.293,4.201,4.187,4.095,4.172,3.976,4.019,3.982,
        3.895,3.760,3.729,3.892,3.859,3.855,3.849,3.838,3.863,3.695,
        3.812,3.821,3.748,3.761,3.676,3.675,3.352,3.404,3.398,3.296,
        3.299,3.247,3.349,3.360,3.401,3.384,3.306,3.271,3.298,3.320,
        3.290,3.130,3.063,3.098,3.099,3.145,3.052,3.056,2.943,2.862,
        2.823,2.828,2.797,2.716,2.756,2.677,2.716,2.745,2.745,2.680,
        2.609,2.628,2.712,2.640,2.807,2.778,2.896,2.889,2.854,2.723,
        2.726,2.706,2.713,2.733,2.732,2.601,2.637,2.592,2.679,2.663,
        2.634,2.559,2.686,2.637,2.697,2.635,2.640,2.599,2.519,2.406,
        2.373,2.342,2.453,2.440,2.500,2.520,2.536,2.502,2.470,2.422,
        2.430,2.506,2.538,2.406,2.236,2.313,2.463,2.423,2.306,2.330,
        2.370,2.406,2.463,2.479,2.306,2.302,2.350,2.343,2.489,2.505,
        2.329,2.422,2.276,2.390,2.409,2.467,2.280,2.470,2.235,2.350,
        2.396,2.467,2.463,2.310,2.299,2.282,2.216,2.297,2.247,2.220,
        2.088,2.175,2.200,2.269,2.320,2.357,2.284,2.273,2.316,2.350,
        2.303,2.357,2.384,2.389,2.353,2.276,2.359,2.259,2.293,2.223,
        2.249,2.297,2.349,2.312,2.303,2.282,2.273,2.316,2.287,2.274,
        2.206,2.142,2.148,2.125,2.048,2.101,2.111,2.200,2.193,2.276,
        2.188,2.197,2.219,2.337,2.359,2.193,2.119,2.085,2.162,2.165,
        2.277,2.226,2.304,2.196,2.155,2.121,2.109,2.162,2.158,2.145,
        2.192,2.229,2.217,2.153,2.004,1.993,1.996,1.989,2.044,2.024,
        2.092,2.116,2.104,2.092,2.089,2.071,2.155,2.175,2.195,2.179,
        2.188,2.179,2.192,2.178,2.158,2.243,2.195,2.106,2.209,2.250,
        2.185,2.240,2.199,2.304,2.316,2.396,2.384,2.306,2.183,2.266,
        2.227,2.399,2.263,2.293,2.180,2.233,2.188,2.252,2.229,2.343,
        2.277,2.273,2.327,2.219,2.166,2.082,2.118,2.111,2.018,1.962,
        2.050,2.041,2.099,2.169,2.153,2.131,2.079,2.079,2.169,2.111,
        2.151,1.980,1.956,1.932,2.018,2.128,2.105,2.227,2.135,2.018,
        1.958,1.905,1.962,2.007,2.220,2.151,2.105,2.070,2.010,2.040,
        2.026,2.121,2.131,2.053,2.033,1.960,1.909,1.928,1.946,1.977,
        2.082,2.018,2.050,1.993,2.070,1.939,2.047,2.129,2.141,2.202,
        2.205,2.172,2.183,2.095,1.977,1.833,1.870,1.907,1.911,1.986,
        1.996,1.972,1.884,1.928,2.131,2.106,2.170,2.099,2.070,1.929,
        2.030,1.891,1.958,2.003,1.955,1.943,1.977,2.044,1.914,1.935,
        1.946,2.091,2.085,2.097,2.077,2.051
    ], dtype=float)

    I_raw_298 = np.array([
        14.45,21.05,22.53,23.59,22.36,21.93,23.44,24.54,25.03,24.92,
        24.81,24.19,23.81,23.54,23.59,23.42,23.07,22.51,22.21,21.89,
        21.57,21.25,21.03,20.70,20.39,19.90,19.38,19.20,18.83,18.60,
        18.43,18.04,17.81,17.45,17.13,16.85,16.53,16.36,16.24,15.99,
        15.69,15.59,15.36,15.31,15.05,14.82,14.38,14.20,13.90,13.83,
        13.72,13.61,13.39,13.17,12.87,12.62,12.49,12.19,12.15,11.89,
        11.79,11.55,11.34,11.11,11.02,11.05,11.06,10.97,10.75,10.50,
        10.42,10.34,10.19,9.952,9.786,9.618,9.744,9.660,9.530,9.267,
        9.170,9.023,8.974,8.828,8.764,8.603,8.359,8.240,8.042,8.160,
        8.130,8.081,7.962,7.849,7.648,7.574,7.403,7.273,7.250,7.255,
        7.127,7.152,6.997,7.001,6.817,6.710,6.602,6.669,6.643,6.573,
        6.495,6.438,6.488,6.444,6.415,6.273,6.269,6.134,6.154,5.989,
        6.073,5.961,5.865,5.767,5.729,5.834,5.868,5.769,5.693,5.671,
        5.580,5.586,5.473,5.414,5.448,5.479,5.532,5.400,5.351,5.176,
        5.115,5.031,5.039,5.058,4.958,4.903,4.910,4.859,4.907,4.957,
        4.995,4.872,4.795,4.673,4.592,4.568,4.573,4.578,4.651,4.618,
        4.548,4.444,4.366,4.354,4.409,4.467,4.399,4.321,4.194,4.333,
        4.343,4.361,4.301,4.288,4.242,4.178,4.078,4.047,4.107,4.020,
        3.993,4.049,4.044,4.121,4.206,4.192,4.171,3.950,3.938,3.854,
        3.880,3.921,3.897,3.847,3.792,3.699,3.800,3.641,3.736,3.651,
        3.668,3.690,3.644,3.639,3.536,3.648,3.639,3.854,3.811,3.804,
        3.708,3.717,3.543,3.540,3.543,3.659,3.697,3.645,3.581,3.426,
        3.440,3.399,3.332,3.370,3.367,3.376,3.429,3.362,3.362,3.405,
        3.327,3.383,3.270,3.317,3.291,3.327,3.315,3.431,3.344,3.297,
        3.277,3.202,3.085,3.096,3.024,3.075,3.232,3.187,3.224,3.256,
        3.205,3.274,3.373,3.347,3.270,3.085,3.233,3.147,3.217,3.119,
        3.042,2.965,2.912,2.896,3.040,2.944,2.952,2.962,2.985,3.175,
        3.262,3.200,3.101,2.849,2.831,2.803,2.848,2.905,2.887,2.955,
        3.057,3.105,3.049,3.092,3.110,3.007,3.003,2.749,2.917,2.882,
        2.915,2.950,2.941,2.843,2.805,2.763,2.830,2.893,2.855,2.825,
        2.836,2.992,2.914,2.967,2.944,2.813,2.900,2.872,2.803,2.810,
        2.809,2.891,2.973,2.896,2.852,2.903,2.791,2.891,2.797,2.776,
        2.722,2.821,2.757,2.807,2.652,2.618,2.591,2.613,2.725,2.612,
        2.667,2.733,2.734,2.703,2.721,2.700,2.706,2.676,2.558,2.540,
        2.540,2.561,2.637,2.588,2.551,2.606,2.688,2.734,2.752,2.821,
        2.659,2.767,2.749,2.799,2.734,2.698,2.745,2.628,2.689,2.689,
        2.707,2.794,2.700,2.727,2.685,2.649,2.763,2.760,2.920,2.763,
        2.713,2.624,2.616,2.645,2.622,2.728,2.595,2.637,2.594,2.646,
        2.722,2.709,2.624,2.576,2.583,2.633,2.622,2.651,2.701,2.624,
        2.628,2.537,2.625,2.566,2.518,2.402,2.563,2.519,2.601,2.504,
        2.640,2.612,2.512,2.435,2.449,2.359,2.418,2.447,2.522,2.464,
        2.424,2.365,2.349,2.494,2.519,2.500,2.377,2.463,2.365,2.377,
        2.378,2.467,2.381,2.390,2.371,2.455,2.574,2.458,2.515,2.411,
        2.512,2.344,2.423,2.429,2.502,2.540,2.613,2.557,2.546,2.552,
        2.568,2.607,2.507,2.506,2.350,2.442
    ], dtype=float)

    I_raw_323 = np.array([
        0.94730,2.219,2.435,2.522,2.534,2.521,2.498,2.472,2.443,2.414,
        2.394,2.367,2.345,2.329,2.307,2.302,2.296,2.274,2.238,2.195,
        2.174,2.160,2.118,2.083,2.038,2.031,2.024,2.004,1.992,1.971,
        1.942,1.903,1.874,1.836,1.829,1.831,1.807,1.796,1.765,1.731,
        1.711,1.704,1.686,1.673,1.646,1.640,1.620,1.612,1.597,1.587,
        1.554,1.548,1.537,1.519,1.482,1.473,1.451,1.426,1.401,1.389,
        1.365,1.360,1.345,1.358,1.332,1.315,1.296,1.276,1.253,1.237,
        1.227,1.208,1.201,1.182,1.173,1.154,1.138,1.139,1.123,1.118,
        1.115,1.100,1.090,1.073,1.065,1.057,1.048,1.042,1.025,1.012,
        9.935,9.994,9.972,9.879,9.840,9.656,9.639,9.421,9.406,9.273,
        9.112,8.973,8.802,8.793,8.873,8.717,8.740,8.668,8.510,8.619,
        8.545,8.346,8.168,8.012,7.922,8.019,7.926,8.045,7.861,7.819,
        7.769,7.767,7.672,7.667,7.602,7.534,7.506,7.407,7.283,7.206,
        7.078,7.217,7.131,7.034,6.980,6.864,6.786,6.831,6.846,6.735,
        6.562,6.553,6.571,6.449,6.318,6.320,6.343,6.370,6.354,6.202,
        6.200,6.148,6.152,6.295,6.295,6.234,6.159,6.022,6.027,6.246,
        6.195,6.193,5.986,5.858,5.719,5.896,5.782,5.919,5.752,5.754,
        5.744,5.765,5.726,5.411,5.580,5.474,5.460,5.479,5.513,5.565,
        5.513,5.555,5.355,5.358,5.276,5.334,5.269,5.332,5.318,5.318,
        5.196,5.213,5.102,4.982,5.036,4.959,5.135,4.994,4.984,4.930,
        4.940,4.888,4.845,4.734,4.698,4.733,4.658,4.707,4.631,4.769,
        4.464,4.514,4.449,4.507,4.553,4.641,4.560,4.615,4.440,4.519,
        4.380,4.425,4.474,4.469,4.332,4.300,4.426,4.351,4.360,4.267,
        4.327,4.370,4.377,4.402,4.262,4.233,4.180,4.247,4.197,4.099,
        3.929,3.873,3.817,3.914,3.912,3.949,4.037,4.054,4.158,4.146,
        4.157,4.102,4.037,3.981,3.887,3.932,3.934,3.836,3.893,3.870,
        3.924,3.925,3.914,3.870,3.799,3.782,3.849,3.873,3.890,3.863,
        3.802,3.476,3.521,3.549,3.640,3.704,3.924,3.934,3.910,3.897,
        3.837,3.797,3.814,3.650,3.741,3.679,3.702,3.788,3.810,3.699,
        3.558,3.439,3.348,3.276,3.228,3.357,3.353,3.541,3.617,3.578,
        3.728,3.643,3.600,3.591,3.549,3.516,3.370,3.398,3.333,3.392,
        3.491,3.494,3.465,3.459,3.457,3.543,3.457,3.417,3.470,3.425,
        3.320,3.347,3.355,3.340,3.270,3.221,3.306,3.358,3.303,3.353,
        3.447,3.377,3.442,3.377,3.445,3.303,3.253,3.293,3.285,3.253,
        3.185,3.280,3.367,3.388,3.201,3.265,3.211,3.298,3.300,3.171,
        3.136,3.090,3.180,3.188,3.338,3.337,3.178,3.250,3.198,3.285,
        3.301,3.221,3.248,3.230,3.141,3.123,3.028,3.063,3.087,3.186,
        3.075,3.093,2.969,3.037,3.087,2.901,2.896,2.625,2.909,2.838,
        2.921,2.970,2.995,2.926,2.813,2.833,2.866,2.886,2.929,2.944,
        2.822,2.826,2.856,2.916,2.785,2.956,2.879,2.876,2.724,2.798,
        2.912,2.858,2.793,2.760,2.731,2.792,2.734,2.817,2.813,2.675,
        2.639,2.706,2.762,2.665,2.650,2.665,2.731,2.764,2.726,2.630,
        2.589,2.545,2.622,2.683,2.736,2.800,2.673,2.749,2.611,2.571,
        2.502,2.602,2.537,2.602,2.604,2.589,2.640,2.548,2.655,2.731,
        2.803,2.785,2.757,2.718,2.670,2.591,
    ], dtype=float)

    I_raw_323[:90] *= 10.0

    # ------------------------------------------------------------------
    # Real experimental cases available from the paper/data extraction.
    # All intensities are converted from the raw percent-like scale to
    # optical-density scale by dividing by 100, exactly as in the 3-block code.
    # ------------------------------------------------------------------
    real_cases = [
        {
            "name": "273K",
            "T": 273.0,
            "is_synthetic": False,
            "xz0": 1.40e-4,
            "xy0": 0.400,
            "xo2": 2.00e-3,
            "singer_k3f_prime": 5.976,
            "singer_k4_prime": 2.592,
            "singer_k2f_prime": 6.720,
            "t_data": t_data.copy(),
            "I_data": I_raw_273 / 100.0,
        },
        {
            "name": "298K",
            "T": 298.0,
            "is_synthetic": False,
            "xz0": 1.40e-4,
            "xy0": 0.400,
            "xo2": 1.90e-3,
            "singer_k3f_prime": 5.998,
            "singer_k4_prime": 3.025,
            "singer_k2f_prime": 6.272,
            "t_data": t_data.copy(),
            "I_data": I_raw_298 / 100.0,
        },
        {
            "name": "323K",
            "T": 323.0,
            "is_synthetic": False,
            "xz0": 1.40e-4,
            "xy0": 0.400,
            "xo2": 1.40e-3,
            "singer_k3f_prime": 6.951,
            "singer_k4_prime": 0.686,
            "singer_k2f_prime": 5.856,
            "t_data": t_data.copy(),
            "I_data": I_raw_323 / 100.0,
        },
    ]

    for case in real_cases:
        if len(case["t_data"]) != len(case["I_data"]):
            raise ValueError(
                "Data mismatch for %s: len(t_data)=%d, len(I_data)=%d"
                % (case["name"], len(case["t_data"]), len(case["I_data"]))
            )

    # ------------------------------------------------------------------
    # Synthetic temperature cases.
    # We interpolate only inside the experimental temperature range
    # [273, 323] K. No extrapolation is allowed.
    #
    # Interpolation details:
    #   * I_data(t): pointwise linear interpolation across temperature
    #     at each fixed time t, using the three available experimental curves.
    #   * xo2, k2f_prime, k3f_prime, k4_prime: scalar linear interpolation
    #     across temperature from the three available values.
    #   * xz0 and xy0: these are constant in the real cases, but they are
    #     still interpolated for consistency.
    #   * K2 and K3 are NOT interpolated; they are recomputed from T below.
    # ------------------------------------------------------------------
    real_T = np.array([case["T"] for case in real_cases], dtype=float)
    real_I_matrix = np.vstack([case["I_data"] for case in real_cases])

    target_temperatures = [273.0, 280.0, 290.0, 298.0, 300.0, 310.0, 320.0, 323.0]
    synthetic_temperatures = {280.0, 290.0, 300.0, 310.0, 320.0}

    def interp_scalar(key, T):
        vals = np.array([case[key] for case in real_cases], dtype=float)
        return float(np.interp(float(T), real_T, vals))

    def interp_curve(T):
        T = float(T)
        return np.array([
            np.interp(T, real_T, real_I_matrix[:, j])
            for j in range(real_I_matrix.shape[1])
        ], dtype=float)

    real_case_by_T = {float(case["T"]): case for case in real_cases}
    cases = []

    for T in target_temperatures:
        T = float(T)
        if T < float(np.min(real_T)) or T > float(np.max(real_T)):
            raise ValueError(
                "Requested temperature %.1f K is outside the interpolation range [%.1f, %.1f] K."
                % (T, float(np.min(real_T)), float(np.max(real_T)))
            )

        if T in real_case_by_T:
            case = dict(real_case_by_T[T])
            case["t_data"] = np.asarray(case["t_data"], dtype=float).copy()
            case["I_data"] = np.asarray(case["I_data"], dtype=float).copy()
        else:
            case = {
                "name": f"{int(round(T))}K",
                "T": T,
                "is_synthetic": True,
                "xz0": interp_scalar("xz0", T),
                "xy0": interp_scalar("xy0", T),
                "xo2": interp_scalar("xo2", T),
                "singer_k3f_prime": interp_scalar("singer_k3f_prime", T),
                "singer_k4_prime": interp_scalar("singer_k4_prime", T),
                "singer_k2f_prime": interp_scalar("singer_k2f_prime", T),
                "t_data": t_data.copy(),
                "I_data": interp_curve(T),
            }

        case["source"] = "synthetic_interpolated" if case.get("is_synthetic", False) else "experimental"
        case["K2"] = 46.0 * np.exp(6500.0 / case["T"] - 18.0)
        case["K3"] = 2.0 * case["K2"]

        if len(case["t_data"]) != len(case["I_data"]):
            raise ValueError(
                "Data mismatch for %s: len(t_data)=%d, len(I_data)=%d"
                % (case["name"], len(case["t_data"]), len(case["I_data"]))
            )

        cases.append(case)

    if len(cases) != 8:
        raise ValueError("Expected 8 temperature cases, got %d." % len(cases))

    return cases


TEMPERATURE_CASES = make_temperature_cases()

# ============================================================
# ODE BLOCK MODEL
# ============================================================

class ODETemperatureBlock:
    def __init__(self, case, block_id=0):
        self.case = dict(case)
        self.block_id = int(block_id)
        self.name = self.case["name"]

    def rhs(self, t, x, k1, k5, k2f, k3f, k4):
        xA, xZ, xY, xD, xB = x

        K2 = float(self.case["K2"])
        K3 = float(self.case["K3"])
        xo2 = float(self.case["xo2"])

        k2r = k2f / K2
        k3r = k3f / K3

        dxA = (
            k1 * xZ * xY
            - xo2 * (k2f + k3f) * xA
            + k2r * xD
            + k3r * xB
            - k5 * xA * xA
        )
        dxZ = -k1 * xZ * xY
        dxY = -k1 * xZ * xY
        dxD = k2f * xA * xo2 - k2r * xD
        dxB = k3f * xo2 * xA - (k3r + k4) * xB

        return [dxA, dxZ, dxY, dxD, dxB]

    def simulate(self, g1, g5, k2f_prime, k3f_prime, k4_prime, rtol=1e-8, atol=1e-10):
        k1 = math.exp(float(g1))
        k5 = math.exp(float(g5))
        k2f = math.exp(float(k2f_prime))
        k3f = math.exp(float(k3f_prime))
        k4 = math.exp(float(k4_prime))

        x0 = [
            0.0,
            float(self.case["xz0"]),
            float(self.case["xy0"]),
            0.0,
            0.0,
        ]
        t_data = np.asarray(self.case["t_data"], dtype=float)

        sol = solve_ivp(
            fun=lambda t, x: self.rhs(t, x, k1, k5, k2f, k3f, k4),
            t_span=(0.0, float(t_data[-1])),
            y0=x0,
            method="BDF",
            t_eval=t_data,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(
                "ODE solve failed for block %s at [g1,g5,k2f_prime,k3f_prime,k4_prime]=[%g,%g,%g,%g,%g]. Message: %s"
                % (self.name, float(g1), float(g5), float(k2f_prime), float(k3f_prime), float(k4_prime), sol.message)
            )

        xA = sol.y[0]
        xD = sol.y[3]
        xB = sol.y[4]

        I_model = 2100.0 * xA + 200.0 * (xB + xD)
        return I_model, sol

    def __call__(self, g1, g5, k2f_prime, k3f_prime, k4_prime):
        try:
            I_model, _ = self.simulate(g1, g5, k2f_prime, k3f_prime, k4_prime)
            I_data = np.asarray(self.case["I_data"], dtype=float)
            val = np.sum((I_data - I_model) ** 2)
        except Exception as exc:
            raise ValueError(
                "Objective evaluation failed for block %s at [g1,g5,k2f_prime,k3f_prime,k4_prime]=[%s,%s,%s,%s,%s]. Error: %s"
                % (self.name, str(g1), str(g5), str(k2f_prime), str(k3f_prime), str(k4_prime), str(exc))
            )

        if not math.isfinite(float(val)):
            raise ValueError(
                "Objective returned non-finite value for block %s at [g1,g5,k2f_prime,k3f_prime,k4_prime]=[%s,%s,%s,%s,%s]: f=%s"
                % (self.name, str(g1), str(g5), str(k2f_prime), str(k3f_prime), str(k4_prime), str(val))
            )

        return float(val)


class ODEStarAssembler:
    def __init__(self, case, block_id=0):
        self.case = dict(case)
        self.block_id = int(block_id)
        self.fixed_k4_prime = float(self.case["singer_k4_prime"])

    def __call__(self, master, private):
        master = np.asarray(master, dtype=float).ravel()
        private = np.asarray(private, dtype=float).ravel()

        if len(master) != 2:
            raise ValueError(f"Block {self.block_id + 1} expects two master values: [ln(k1), ln(k5)].")
        if len(private) != 2:
            raise ValueError(
                f"Block {self.block_id + 1} expects private dimension 2, got {len(private)}."
            )

        # k2f_prime and k3f_prime are optimized; k4_prime is fixed.
        return tuple([
            float(master[0]),
            float(master[1]),
            float(private[0]),
            float(private[1]),
            float(self.fixed_k4_prime),
        ])


# ============================================================
# DBDDSBB PROBLEM
# ============================================================
def build_problem():
    """Build the 18D star-decomposed ODE parameter-estimation problem."""

    if len(TEMPERATURE_CASES) != N_BLOCKS:
        raise ValueError(
            f"Expected {N_BLOCKS} temperature cases, got {len(TEMPERATURE_CASES)}."
        )

    block_specs = []

    for b, case in enumerate(TEMPERATURE_CASES):
        private_start = MASTER_DIM + b * PRIVATE_DIM
        private_indices = [private_start, private_start + 1]

        block_specs.append({
            "name": (
                f"J_{case['name']}"
                "(ln_k1,ln_k5,k2f_prime,k3f_prime;fixed_k4_prime)"
            ),
            "temperature_name": case["name"],
            "master_indices": [0, 1],
            "private_names": [
                f"k2f_prime_{case['name']}",
                f"k3f_prime_{case['name']}",
            ],
            "private_original_indices": private_indices,
            "private_bounds": [
                (K2F_PRIME_LB, K2F_PRIME_UB),
                (K3F_PRIME_LB, K3F_PRIME_UB),
            ],
            "fixed_k4_prime": float(case["singer_k4_prime"]),
            "block_function": ODETemperatureBlock(case=case, block_id=b),
            "assemble_args": ODEStarAssembler(case=case, block_id=b),
        })

    evaluator = DecomposedProblemEvaluator(
        block_specs=block_specs,
        master_dim=MASTER_DIM,
        full_dim=FULL_DIM,
        m0=M_PRIVATE_0,
        gamma=M_PRIVATE_GAMMA,
        level_interval=M_PRIVATE_LEVEL_INTERVAL,
        sample_per_axis=SAMPLE_PER_AXIS,
        parallel_function_evaluations=PARALLEL_FUNCTION_EVALUATIONS,
    )

    problem = DBDDSBB.Problem()
    problem.add_objective(evaluator.objective, sense="minimize")

    # Shared/master variables: ln(k1) and ln(k5).
    problem.add_variable(G1_LB, G1_UB)
    problem.add_variable(G5_LB, G5_UB)

    evaluator.attach_to_problem(problem)

    return problem, evaluator, block_specs


# ============================================================
# DBDDSBB SOLVER
# ============================================================
def build_solver(time_limit):
    return DBDDSBB.DBDDSBB(
        50,
        formulation=FORMULATION,
        split_method="equal_bisection",
        variable_selection="longest_side",
        stop_option={
            "absolute_tolerance": 1.0e-7,
            "relative_tolerance": 1.0e-8,
            "minimum_bound": 1.0e-9,
            "sampling_limit": 10**30,
            "max_feval_total_limit": 10**30,
            "max_feval_equivalent_limit": 10**30,
            "time_limit": float(time_limit),
        },
    )


# ============================================================
# FITTED TRAJECTORY OUTPUT
# ============================================================
def write_fit_csvs(x_full, block_specs, time_limit):
    if x_full is None:
        return

    x_full = np.asarray(x_full, dtype=float).ravel()
    if len(x_full) != FULL_DIM:
        return

    g1 = float(x_full[0])
    g5 = float(x_full[1])

    for b, spec in enumerate(block_specs):
        case = TEMPERATURE_CASES[b]
        block = spec["block_function"]
        private_idx = spec["private_original_indices"]

        k2f_prime = float(x_full[private_idx[0]])
        k3f_prime = float(x_full[private_idx[1]])
        k4_prime = float(case["singer_k4_prime"])

        I_model, sol = block.simulate(
            g1,
            g5,
            k2f_prime,
            k3f_prime,
            k4_prime,
        )

        csv_path = (
            f"ode_{case['name']}_star_18d_{FORMULATION}"
            f"_fit_tlim_{float(time_limit):g}s.csv"
        )

        header = "time_microseconds,I_data,I_model,xA,xZ,xY,xD,xB"
        csv_data = np.column_stack([
            case["t_data"],
            case["I_data"],
            I_model,
            sol.y[0],
            sol.y[1],
            sol.y[2],
            sol.y[3],
            sol.y[4],
        ])

        np.savetxt(
            csv_path,
            csv_data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.12e",
        )

        print(f"Saved fitted trajectory CSV for {case['name']}: {csv_path}")


# ============================================================
# RUN ONE TIME-LIMIT CASE
# ============================================================
def run_single_case(time_limit):
    random.seed(SEED)
    np.random.seed(SEED)

    problem, evaluator, block_specs = build_problem()
    solver = build_solver(time_limit)

    print("=" * 100)
    print(
        f"EIGHT-TEMPERATURE STAR-STRUCTURED ODE PARAMETER ESTIMATION "
        f"- 18D - {FORMULATION}"
    )
    print("=" * 100)
    print(f"TIME LIMIT: {float(time_limit)} s")
    print(f"MPI SIZE: {SIZE}")
    print(f"PARALLEL FUNCTION EVALUATIONS: {PARALLEL_FUNCTION_EVALUATIONS}")
    print("=" * 100)

    solver.optimize(problem)
    solver.print_result()
    solver._update_blockwise_feval_accounting()

    yopt = solver.get_optimum()

    try:
        xopt_full = solver.get_optimizer()
    except Exception:
        xopt_full = None

    gap = solver.yopt_global - solver.lowerbound_global

    print("Best sampled feasible objective value:", yopt)
    print("Best stored full minimizer:", xopt_full)

    if xopt_full is not None:
        xopt_arr = np.asarray(xopt_full, dtype=float).ravel()

        if len(xopt_arr) == FULL_DIM:
            print("Best physical variables:")
            print("  k1 = %.10f" % math.exp(float(xopt_arr[0])))
            print("  k5 = %.10f" % math.exp(float(xopt_arr[1])))

            for b, spec in enumerate(block_specs):
                private_idx = spec["private_original_indices"]
                temp_name = spec["temperature_name"]
                fixed_k4_prime = float(TEMPERATURE_CASES[b]["singer_k4_prime"])

                print(
                    "  k2f_%s = %.10f"
                    % (temp_name, math.exp(float(xopt_arr[private_idx[0]])))
                )
                print(
                    "  k3f_%s = %.10f"
                    % (temp_name, math.exp(float(xopt_arr[private_idx[1]])))
                )
                print(
                    "  k4_%s  = %.10f (fixed, k4_prime = %.10f)"
                    % (
                        temp_name,
                        math.exp(fixed_k4_prime),
                        fixed_k4_prime,
                    )
                )

            if SAVE_BEST_FIT_CSVS:
                write_fit_csvs(
                    xopt_arr,
                    block_specs,
                    time_limit=time_limit,
                )

    return {
        "time_limit": float(time_limit),
        "n_blocks": N_BLOCKS,
        "master_dim": MASTER_DIM,
        "private_dim": PRIVATE_DIM,
        "full_dim": FULL_DIM,
        "elapsed_time": float(solver.time_total),
        "samples_used": int(solver.builder.simulator.sample_number),
        "block_fevals_total": solver.total_blockwise_fevals,
        "block_fevals_equiv": solver.equivalent_blockwise_fevals,
        "LB": float(solver.lowerbound_global),
        "UB": float(solver.yopt_global),
        "abs_gap": float(gap) if np.isfinite(gap) else np.nan,
        "level": int(solver.level),
        "node": int(solver.builder.node),
        "active_m_private": int(evaluator.current_m),
        "yopt": float(yopt),
    }


# ============================================================
# RESULTS
# ============================================================
def write_results_csv(results):
    fieldnames = [
        "time_limit",
        "n_blocks",
        "master_dim",
        "private_dim",
        "full_dim",
        "elapsed_time",
        "samples_used",
        "block_fevals_total",
        "block_fevals_equiv",
        "LB",
        "UB",
        "abs_gap",
        "level",
        "node",
        "active_m_private",
        "yopt",
    ]

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_controller():
    results = []

    for time_limit in TIME_LIMITS:
        print("=" * 100)
        print(f"STARTING TIME-LIMIT SWEEP CASE: {float(time_limit):.6f} s")
        print("=" * 100)

        results.append(run_single_case(time_limit))

    print("=" * 100)
    print(f"ODE TIME-SWEEP SUMMARY - {FORMULATION}")
    print("=" * 100)

    for r in results:
        print(
            f"TIME_LIMIT={r['time_limit']:>8.2f} s, "
            f"FULL_DIM={r['full_dim']:>3d}, "
            f"TIME={r['elapsed_time']:.6f} s, "
            f"EQ_FE={r['block_fevals_equiv']}, "
            f"LEVEL={r['level']}, "
            f"NODE={r['node']}, "
            f"M_PRIVATE={r['active_m_private']}, "
            f"LB={r['LB']}, "
            f"UB={r['UB']}, "
            f"GAP={r['abs_gap']}"
        )

    write_results_csv(results)

    print("=" * 100)
    print(f"Saved time-sweep CSV results to: {CSV_OUTPUT}")

    if COMM is not None and SIZE > 1:
        DBDDSBB._DB_underestimator.stop_two_level_workers()


# ============================================================
# MPI WORKERS
# ============================================================
def run_worker():
    if COMM is not None and SIZE > 1:
        DBDDSBB._DB_underestimator.run_two_level_worker_loop()


def main():
    if RANK == 0:
        run_controller()
    else:
        run_worker()


if __name__ == "__main__":
    main()
