#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
import numpy as np
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


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
SEED = 12345

# Ten subject blocks. Each block has two shared parameters and one private
# random effect, so the full problem dimension is D = 2 + 10 = 12.
N_BLOCKS = 10
MASTER_DIM = 2
PRIVATE_DIM = 1
BLOCK_DIM = MASTER_DIM + PRIVATE_DIM
FULL_DIM = MASTER_DIM + N_BLOCKS * PRIVATE_DIM

# Shared variables: theta = [theta1_log, theta2_log].
THETA1_LB, THETA1_UB = np.log(0.1), np.log(2.0)
THETA2_LB, THETA2_UB = np.log(0.05), np.log(1.0)

# Private variable for each subject: b_i.
B_LB, B_UB = -2.0, 2.0

# Lower-bound formulation: choose "F1" or "F2".
FORMULATION = "F2"

# Private-partition refinement:
# M_nu = ceil(M0 * gamma^floor((nu - 1) / L_M)).
# The solver tree level is nu - 1.
#
# The previous architecture used M_PRIVATE = 30. Thus M0 = 30 preserves
# the initial private-grid resolution, while allowing refinement with level.
M_PRIVATE_0 = 30
M_PRIVATE_GAMMA = 1.2
M_PRIVATE_LEVEL_INTERVAL = 15
SAMPLE_PER_AXIS = 2

# F2 problem-level parameters.
F2_GAMMA0 = 2.5
F2_RHO = 2.5
F2_OMEGA = 0.001

# The analytic block evaluations are inexpensive.
PARALLEL_FUNCTION_EVALUATIONS = False

# Solve the same 12D problem once for each time limit (seconds).
TIME_LIMITS = [100, 500, 1000, 2000]

BLOCK_FEVAL_EQUIV_LIMIT = 10**15
CSV_OUTPUT = f"nlme_ode_time_sweep_results_{FORMULATION}.csv"


random.seed(SEED)
np.random.seed(SEED)


# Configure F2 parameters. These are used when FORMULATION = "F2".
DBDDSBB._DB_underestimator.configure_formulation2(
    gamma0=F2_GAMMA0,
    rho=F2_RHO,
    omega=F2_OMEGA,
)


# -----------------------------------------------------------------------------
# Known-variance NLME-ODE problem
# -----------------------------------------------------------------------------
# True data-generation parameters.
TRUE_THETA1_LOG = np.log(0.5)
TRUE_THETA2_LOG = np.log(0.2)
TRUE_SIGMA = 0.05
TRUE_SIGMA_B = 0.5
B_PENALTY_COEFF = (TRUE_SIGMA**2) / (TRUE_SIGMA_B**2)

X0_MEAN = np.array([2.0, 3.0], dtype=float)
TIMES_OBS = np.linspace(0.0, 10.0, 11)


def lin_ode_solution_x1(times, theta1_log, theta2_log, b_i, x0_i):
    """Analytic X1 solution of the two-state linear ODE model."""

    t = np.asarray(times, dtype=float).ravel()
    x0_i = np.asarray(x0_i, dtype=float).ravel()

    x10 = float(x0_i[0])
    x20 = float(x0_i[1])

    phi1 = float(np.exp(theta1_log + b_i))
    phi2 = float(np.exp(theta2_log))

    if abs(phi1 - phi2) < 1.0e-10:
        x1 = np.exp(-phi2 * t) * (x10 + phi2 * x20 * t)
    else:
        x1 = (
            x10 * np.exp(-phi1 * t)
            + (phi2 * x20 / (phi1 - phi2))
            * (np.exp(-phi2 * t) - np.exp(-phi1 * t))
        )

    return np.asarray(x1, dtype=float)


def generate_linear_nlme_data(n_subjects, seed=SEED):
    """Generate the fixed synthetic subject data used by the benchmark."""

    rng = np.random.default_rng(seed)
    subjects = []

    for _ in range(int(n_subjects)):
        btrue = TRUE_SIGMA_B * rng.normal(0.0, 1.0)

        z0 = rng.normal(0.0, 1.0)
        x0_i = np.maximum(X0_MEAN * (1.0 + 0.5 * z0), 0.5)

        x1_clean = lin_ode_solution_x1(
            TIMES_OBS,
            TRUE_THETA1_LOG,
            TRUE_THETA2_LOG,
            btrue,
            x0_i,
        )

        y_i = (
            x1_clean
            + TRUE_SIGMA * rng.normal(0.0, 1.0, size=len(TIMES_OBS))
        )

        subjects.append({
            "times": TIMES_OBS.copy(),
            "y": np.asarray(y_i, dtype=float).copy(),
            "x0": np.asarray(x0_i, dtype=float).copy(),
            "btrue": float(btrue),
        })

    return subjects


SUBJECT_DATA = generate_linear_nlme_data(N_BLOCKS, seed=SEED)


class LinearNLMEODEBlock:
    """
    Known-variance subject-level objective:

        F_i(theta1_log, theta2_log, b_i)
        =
        sum_j (y_ij - X1_i(t_j))^2
        + (sigma^2 / sigma_b^2) b_i^2.
    """

    def __init__(self, subject_data, block_id):
        self.subject_data = subject_data
        self.block_id = int(block_id)

    def __call__(self, theta1_log, theta2_log, b_i):
        theta1_log = float(theta1_log)
        theta2_log = float(theta2_log)
        b_i = float(b_i)

        x1_pred = lin_ode_solution_x1(
            times=self.subject_data["times"],
            theta1_log=theta1_log,
            theta2_log=theta2_log,
            b_i=b_i,
            x0_i=self.subject_data["x0"],
        )

        residual = self.subject_data["y"] - x1_pred
        sse_term = np.sum(residual**2)
        random_effect_penalty = B_PENALTY_COEFF * b_i**2

        return float(sse_term + random_effect_penalty)


class LinearNLMEODEAssembler:
    """Map two shared parameters and one private random effect to a block."""

    def __call__(self, master, private):
        master = np.asarray(master, dtype=float).ravel()
        private = np.asarray(private, dtype=float).ravel()

        if len(master) != MASTER_DIM or len(private) != PRIVATE_DIM:
            raise ValueError(
                "Each NLME block requires two master variables "
                "and one private variable."
            )

        return (
            float(master[0]),
            float(master[1]),
            float(private[0]),
        )


def build_problem():
    """Build the 12D, 10-subject star-decomposed NLME-ODE problem."""

    block_specs = []

    for b in range(N_BLOCKS):
        private_index = MASTER_DIM + b

        block_specs.append({
            "name": f"subject_{b + 1}",
            "master_indices": [0, 1],
            "private_names": [f"b_{b + 1}"],
            "private_original_indices": [private_index],
            "private_bounds": [(B_LB, B_UB)],
            "block_function": LinearNLMEODEBlock(
                subject_data=SUBJECT_DATA[b],
                block_id=b,
            ),
            "assemble_args": LinearNLMEODEAssembler(),
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

    # Shared/master variables: theta1_log and theta2_log.
    problem.add_variable(THETA1_LB, THETA1_UB)
    problem.add_variable(THETA2_LB, THETA2_UB)

    evaluator.attach_to_problem(problem)

    return problem, evaluator


# -----------------------------------------------------------------------------
# DBDDSBB solver
# -----------------------------------------------------------------------------
def build_solver(time_limit):
    return DBDDSBB.DBDDSBB(
        50,
        formulation=FORMULATION,
        split_method="equal_bisection",
        variable_selection="longest_side",
        stop_option={
            "absolute_tolerance": 0.1,
            "relative_tolerance": 1.0e-8,
            "minimum_bound": 1.0e-9,
            "sampling_limit": 10**30,
            "max_feval_total_limit": 10**30,
            "max_feval_equivalent_limit": BLOCK_FEVAL_EQUIV_LIMIT,
            "time_limit": float(time_limit),
        },
    )


# -----------------------------------------------------------------------------
# Run one time-limit case
# -----------------------------------------------------------------------------
def run_single_case(time_limit):
    # Reset the random state so every time-limit case is reproducible.
    random.seed(SEED)
    np.random.seed(SEED)

    problem, evaluator = build_problem()
    solver = build_solver(time_limit)

    solver.optimize(problem)
    solver.print_result()
    solver._update_blockwise_feval_accounting()

    gap = solver.yopt_global - solver.lowerbound_global

    try:
        xopt_full = solver.get_optimizer()
    except Exception:
        xopt_full = None

    if xopt_full is not None:
        xopt_arr = np.asarray(xopt_full, dtype=float).ravel()
        if len(xopt_arr) == FULL_DIM:
            print("Estimated shared variables:")
            print("  theta1_log =", xopt_arr[0])
            print("  theta2_log =", xopt_arr[1])

    return {
        "time_limit": float(time_limit),
        "block_fevals_equiv_limit": int(BLOCK_FEVAL_EQUIV_LIMIT),
        "n_blocks": N_BLOCKS,
        "block_dim": BLOCK_DIM,
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
        "yopt": float(solver.get_optimum()),
    }


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
def write_results_csv(results):
    fieldnames = [
        "time_limit",
        "block_fevals_equiv_limit",
        "n_blocks",
        "block_dim",
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
    results = [
        run_single_case(time_limit)
        for time_limit in TIME_LIMITS
    ]

    print("=" * 100)
    print(f"NLME-ODE TIME-LIMIT SWEEP SUMMARY - {FORMULATION}")
    print("=" * 100)

    for r in results:
        print(
            f"TIME_LIMIT={r['time_limit']:>7.1f} s, "
            f"FULL_DIM={r['full_dim']:>3d}, "
            f"TIME={r['elapsed_time']:.6f} s, "
            f"EQ_FE={r['block_fevals_equiv']}, "
            f"LEVEL={r['level']}, "
            f"NODE={r['node']}, "
            f"M_PRIVATE={r['active_m_private']}, "
            f"LB={r['LB']}, "
            f"UB={r['UB']}"
        )

    write_results_csv(results)

    print("=" * 100)
    print(f"Saved CSV results to: {CSV_OUTPUT}")

    if COMM is not None and SIZE > 1:
        DBDDSBB._DB_underestimator.stop_two_level_workers()


# -----------------------------------------------------------------------------
# MPI workers
# -----------------------------------------------------------------------------
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
