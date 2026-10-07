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

# Number of star blocks; corresponds to full dimensions D = [3, 4, 10, 20, 50].
BLOCKS_LIST = [2, 3, 9, 19, 49]

# Search bounds: s is the shared/master variable x_D, while each private
# variable x_i belongs to its corresponding star block.
S_LB, S_UB = -2.0, 2.0
PRIVATE_LB, PRIVATE_UB = -2.0, 2.0

# Lower-bound formulation: choose "F1" or "F2".
FORMULATION = "F1"

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

# Parallelize private-grid function evaluations across MPI workers.
PARALLEL_FUNCTION_EVALUATIONS = False

CSV_OUTPUT = "arrowhead_scaling_results.csv"


random.seed(SEED)
np.random.seed(SEED)


# Configure F2 parameters. These are used when FORMULATION = "F2".
DBDDSBB._DB_underestimator.configure_formulation2(
    gamma0=F2_GAMMA0,
    rho=F2_RHO,
    omega=F2_OMEGA,
)


# -----------------------------------------------------------------------------
# ARWHEAD decomposition
# -----------------------------------------------------------------------------
class ARWHEADBlock:
    """One star block: shared x_D plus one private x_i."""

    def __init__(self, block_id):
        self.block_id = int(block_id)

    def __call__(self, xn, xi):
        xn = float(xn)
        xi = float(xi)

        return float(
            (xi**2 + xn**2)**2
            - 4.0 * xi
            + 3.0
        )


class StarAssembler:
    """Map [master], [private] to the arguments of ARWHEADBlock."""

    def __call__(self, master, private):
        master = np.asarray(master, dtype=float).ravel()
        private = np.asarray(private, dtype=float).ravel()

        if len(master) != 1 or len(private) != 1:
            raise ValueError(
                "ARWHEAD blocks require one master and one private variable."
            )

        return float(master[0]), float(private[0])


def build_problem(n_blocks):
    """Build the star-decomposed ARWHEAD problem."""

    n_blocks = int(n_blocks)
    full_dim = n_blocks + 1

    block_specs = []

    for b in range(n_blocks):
        private_index = b + 1

        block_specs.append({
            "name": f"g{b + 1}(s)",
            "master_indices": [0],
            "private_names": [f"x{private_index + 1}"],
            "private_original_indices": [private_index],
            "private_bounds": [(PRIVATE_LB, PRIVATE_UB)],
            "block_function": ARWHEADBlock(b),
            "assemble_args": StarAssembler(),
        })

    evaluator = DecomposedProblemEvaluator(
        block_specs=block_specs,
        master_dim=1,
        full_dim=full_dim,
        m0=M_PRIVATE_0,
        gamma=M_PRIVATE_GAMMA,
        level_interval=M_PRIVATE_LEVEL_INTERVAL,
        sample_per_axis=SAMPLE_PER_AXIS,
        parallel_function_evaluations=PARALLEL_FUNCTION_EVALUATIONS,
    )

    problem = DBDDSBB.Problem()

    problem.add_objective(
        evaluator.objective,
        sense="minimize",
    )

    # Shared/master variable s = x_D.
    problem.add_variable(S_LB, S_UB)

    evaluator.attach_to_problem(problem)

    return problem, full_dim


# -----------------------------------------------------------------------------
# DBDDSBB solver
# -----------------------------------------------------------------------------
def build_solver():
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
            "max_feval_equivalent_limit": 10**30,
            "time_limit": 3600,
        },
    )


# -----------------------------------------------------------------------------
# Run one ARWHEAD case
# -----------------------------------------------------------------------------
def run_single_case(n_blocks):

    # Reset the random state so every scaling case is reproducible.
    random.seed(SEED)
    np.random.seed(SEED)

    problem, full_dim = build_problem(n_blocks)
    solver = build_solver()

    solver.optimize(problem)
    solver.print_result()

    solver._update_blockwise_feval_accounting()

    gap = solver.yopt_global - solver.lowerbound_global

    return {
        "n_blocks": int(n_blocks),
        "block_dim": 2,
        "private_dim": 1,
        "full_dim": int(full_dim),
        "elapsed_time": float(solver.time_total),
        "samples_used": int(solver.builder.simulator.sample_number),
        "block_fevals_total": solver.total_blockwise_fevals,
        "block_fevals_equiv": solver.equivalent_blockwise_fevals,
        "LB": float(solver.lowerbound_global),
        "UB": float(solver.yopt_global),
        "abs_gap": float(gap) if np.isfinite(gap) else np.nan,
        "level": int(solver.level),
        "node": int(solver.builder.node),
        "yopt": float(solver.get_optimum()),
    }


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
def write_results_csv(results):

    fieldnames = [
        "n_blocks",
        "block_dim",
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
        "yopt",
    ]

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def run_controller():

    results = [
        run_single_case(n_blocks)
        for n_blocks in BLOCKS_LIST
    ]

    print("=" * 100)
    print(f"ARWHEAD SCALING SUMMARY - {FORMULATION}")
    print("=" * 100)

    for r in results:
        print(
            f"N_BLOCKS={r['n_blocks']:>3d}, "
            f"FULL_DIM={r['full_dim']:>4d}, "
            f"TIME={r['elapsed_time']:.6f} s, "
            f"EQ_FE={r['block_fevals_equiv']}, "
            f"LEVEL={r['level']}, "
            f"NODE={r['node']}, "
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
