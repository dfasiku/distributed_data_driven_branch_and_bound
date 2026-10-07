#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
DBDDSBB was developed using the publicly available PyDDSBB implementation
as a starting codebase. The original PyDDSBB implementation was substantially
modified and extended to implement the proposed decomposition-based framework.

Original PyDDSBB implementation:
https://github.com/DDPSE/PyDDSBB
"""

import pyomo.environ as pe
import numpy as np
import time
import itertools
import heapq
import hashlib
from scipy.spatial import ConvexHull, QhullError
import os

INFINITY = np.inf


# ============================================================================
# FORMULATION 2 SETTINGS
# ============================================================================
# Geometry / residual correction
GEOMETRY_THRESHOLD = 2.0          # Gamma_0 default; call file may override
RHO_RESIDUAL = 2.0                 # rho default; call file may override
OMEGA_RESIDUAL = 0.001             # omega > 0; call file may override
RESIDUAL_NEIGHBORS = 5             # minimum nearest-neighbor count

# IID sampling policy used by DDSBB for Formulation 2
IID_TARGET_MULTIPLIER = 3         # target total = multiplier * (2*d_s + 1)
IID_MIN_FRESH_PER_NODE = 4        # minimum fresh IID samples at every child node
IID_GEOMETRY_BATCH_MULTIPLIER = 1 # enrichment batch = multiplier * d_s
MAX_GEOMETRY_ENRICH_SAMPLES = 20  # maximum extra IID samples for geometry repair

# Certified multidimensional coupling-space fill distance
FILL_DISTANCE_ABS_TOL = 1.0e-3
FILL_DISTANCE_REL_TOL = 1.0e-2
FILL_DISTANCE_MAX_CELLS = 10000


def configure_formulation2(*, gamma0, rho, omega):
    """Set the problem-level Formulation-2 parameters.

    The remaining F2 sampling, geometry-enrichment, and fill-distance settings
    are implementation defaults defined in this module. Call this on every MPI
    rank before constructing/entering the DBDDSBB solver.
    """
    global GEOMETRY_THRESHOLD, RHO_RESIDUAL, OMEGA_RESIDUAL

    gamma0 = float(gamma0)
    rho = float(rho)
    omega = float(omega)
    if gamma0 <= 0.0:
        raise ValueError("DBDDSBB F2: gamma0 must be positive.")
    if rho <= 0.0:
        raise ValueError("DBDDSBB F2: rho must be positive.")
    if omega <= 0.0:
        raise ValueError("DBDDSBB F2: omega must be positive.")

    GEOMETRY_THRESHOLD = gamma0
    RHO_RESIDUAL = rho
    OMEGA_RESIDUAL = omega


class GeometryEnrichmentRequired(RuntimeError):
    """Signal DDSBB to add fresh IID coupling samples and refit the node."""

    def __init__(self, diagnostics):
        self.diagnostics = list(diagnostics)
        lines = ["TwoLevelQuadratic: sampling geometry is not acceptable."]
        for rec in self.diagnostics[:10]:
            lines.append(
                "  block={block_name}, region={region_index}, "
                "kappa={kappa:.6g}, Gamma0={Gamma0:.6g}, "
                "dN={dN:.6g}, h_lower={h_lower:.6g}, h_upper={h_upper:.6g}, "
                "dN/(h_lower)={ratio:.6g}".format(**rec)
            )
        if len(self.diagnostics) > 10:
            lines.append(f"  ... and {len(self.diagnostics) - 10} additional failing regions.")
        super().__init__("\n".join(lines))

#============================================================
# MPI HELPERS FOR GENERALIZED TWO-LEVEL QUADRATIC
# ============================================================

_MPI = None

def _get_mpi():
    """Import mpi4py only when MPI execution is actually requested."""
    global _MPI
    if _MPI is None:
        try:
            from mpi4py import MPI as mpi
        except Exception as exc:
            raise RuntimeError("DBDDSBB: MPI was requested but mpi4py/MPI is unavailable.") from exc
        _MPI = mpi
    return _MPI

def _parallel_mpi_if_active():
    """Return MPI only inside a multi-process launcher; otherwise stay serial."""
    size_vars = ("PMI_SIZE", "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS")
    sizes = []
    for name in size_vars:
        try:
            sizes.append(int(os.environ.get(name, "1")))
        except ValueError:
            pass
    if max(sizes or [1]) <= 1:
        return None
    return _get_mpi()

_TLQ_WORKER_INSTANCE = None


def _make_two_level_worker_instance(intercept=True, solver="highs", sample_per_axis=5):
    return TwoLevelQuadratic(
        intercept=intercept,
        solver=solver,
        sample_per_axis=sample_per_axis,
    )


def _split_indices_static(n_items, size):
    """Round-robin split of integer indices across MPI ranks."""
    chunks = [[] for _ in range(int(size))]
    for i in range(int(n_items)):
        chunks[i % int(size)].append(int(i))
    return chunks


def _serial_eval_private_indices(master_values, block_function, assemble_args, private_points, indices):
    """Evaluate selected private-grid points and retain original indices."""
    mvals = np.asarray(master_values, dtype=float).ravel()
    private_points = np.asarray(private_points, dtype=float)
    if private_points.ndim != 2:
        raise ValueError("DBDDSBB: private_points must be a 2D array.")

    local_results = []
    for i in indices:
        i = int(i)
        pvals = private_points[i]
        args = assemble_args(mvals, pvals)
        val = float(block_function(*args))
        local_results.append((i, val))

    return {"results": local_results, "n_eval": int(len(local_results))}


def evaluate_private_grid(master_values, block_function, assemble_args, private_grid, *, parallel_function_evaluations=False):
    """Evaluate a block over its private grid, optionally distributing the
    actual function evaluations across the persistent MPI workers.

    The returned block_values always preserve the original private-grid order.
    n_eval counts mathematical block-function evaluations, not MPI/Python calls.
    """
    mvals = np.asarray(master_values, dtype=float).ravel()
    private_points = np.asarray(private_grid, dtype=float)
    if private_points.ndim != 2:
        raise ValueError("DBDDSBB: private_grid must be a 2D array.")

    n_private_points = int(private_points.shape[0])
    if n_private_points <= 0:
        raise ValueError("DBDDSBB: private_grid must contain at least one point.")

    MPI = _parallel_mpi_if_active() if bool(parallel_function_evaluations) else None

    if MPI is None or MPI.COMM_WORLD.Get_size() <= 1:
        payload = _serial_eval_private_indices(
            mvals, block_function, assemble_args, private_points,
            range(n_private_points),
        )
        vals = np.empty(n_private_points, dtype=float)
        for i, val in payload["results"]:
            vals[int(i)] = float(val)
        total_eval = int(payload["n_eval"])
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank != 0:
            raise RuntimeError(
                "DBDDSBB: evaluate_private_grid() may only be called by rank 0. "
                "Worker ranks must remain inside run_two_level_worker_loop()."
            )

        tasks_by_rank = _split_indices_static(n_private_points, size)
        msg = {
            "cmd": "evaluate_private_grid",
            "config": {
                "master_values": np.asarray(mvals, dtype=float),
                "private_points": np.asarray(private_points, dtype=float),
                "block_function": block_function,
                "assemble_args": assemble_args,
            },
            "tasks_by_rank": tasks_by_rank,
        }
        comm.bcast(msg, root=0)

        local_payload = _serial_eval_private_indices(
            mvals, block_function, assemble_args, private_points,
            tasks_by_rank[0],
        )
        gathered_payloads = comm.gather(local_payload, root=0)

        vals = np.full(n_private_points, np.nan, dtype=float)
        total_eval = 0
        for payload in gathered_payloads:
            if payload is None:
                continue
            total_eval += int(payload.get("n_eval", 0))
            for i, val in payload["results"]:
                vals[int(i)] = float(val)

        if total_eval != n_private_points:
            raise RuntimeError(
                "DBDDSBB: parallel private-grid evaluation returned "
                f"{total_eval} evaluations; expected {n_private_points}."
            )

    if np.any(~np.isfinite(vals)):
        bad = np.where(~np.isfinite(vals))[0]
        raise RuntimeError(
            "DBDDSBB: private-grid evaluation returned non-finite or missing "
            f"values at indices {bad[:20].tolist()}."
        )

    j = int(np.argmin(vals))
    return {
        "best_val": float(vals[j]),
        "best_private": np.asarray(private_points[j], dtype=float).copy(),
        "private_points": private_points.copy(),
        "block_values": vals.copy(),
        "n_eval": int(total_eval),
    }


def run_two_level_worker_loop():
    MPI = _get_mpi()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size <= 1 or rank == 0:
        return

    global _TLQ_WORKER_INSTANCE

    while True:
        msg = comm.bcast(None, root=0)

        if not isinstance(msg, dict) or "cmd" not in msg:
            raise RuntimeError("TwoLevelQuadratic worker received malformed command.")

        cmd = msg["cmd"]

        if cmd == "stop":
            break

        # Actual expensive block-function evaluations over a private grid.
        # Handle this before the TLQ-specific worker configuration below.
        if cmd == "evaluate_private_grid":
            cfg = msg.get("config", None)
            tasks_by_rank = msg.get("tasks_by_rank", None)
            if not isinstance(cfg, dict):
                raise RuntimeError("DBDDSBB worker received malformed evaluate_private_grid config.")
            if not isinstance(tasks_by_rank, (list, tuple)):
                raise RuntimeError("DBDDSBB worker received malformed evaluate_private_grid tasks_by_rank.")
            if rank >= len(tasks_by_rank):
                raise RuntimeError("DBDDSBB worker rank exceeds evaluate_private_grid tasks_by_rank length.")

            required = ("master_values", "private_points", "block_function", "assemble_args")
            missing = [key for key in required if key not in cfg]
            if missing:
                raise RuntimeError(
                    "DBDDSBB worker evaluate_private_grid config is missing "
                    f"keys {missing}."
                )

            local_payload = _serial_eval_private_indices(
                master_values=np.asarray(cfg["master_values"], dtype=float),
                block_function=cfg["block_function"],
                assemble_args=cfg["assemble_args"],
                private_points=np.asarray(cfg["private_points"], dtype=float),
                indices=tasks_by_rank[rank],
            )
            comm.gather(local_payload, root=0)
            continue

        cfg = msg.get("config", None)
        if not isinstance(cfg, dict):
            raise RuntimeError("TwoLevelQuadratic worker received malformed config.")

        if _TLQ_WORKER_INSTANCE is None:
            _TLQ_WORKER_INSTANCE = _make_two_level_worker_instance(
                intercept=cfg["intercept"],
                solver=cfg["solver_name"],
                sample_per_axis=cfg["sample_per_axis"],
            )
        else:
            _TLQ_WORKER_INSTANCE.intercept = bool(cfg["intercept"])
            _TLQ_WORKER_INSTANCE.sample_per_axis = int(max(1, cfg["sample_per_axis"]))
            _TLQ_WORKER_INSTANCE.solver_name = str(cfg["solver_name"])
            _TLQ_WORKER_INSTANCE.solver = pe.SolverFactory(_TLQ_WORKER_INSTANCE.solver_name)

            if (
                _TLQ_WORKER_INSTANCE.solver is None
                or not _TLQ_WORKER_INSTANCE.solver.available(exception_flag=False)
            ):
                _TLQ_WORKER_INSTANCE.solver_name = "highs"
                _TLQ_WORKER_INSTANCE.solver = pe.SolverFactory(_TLQ_WORKER_INSTANCE.solver_name)

            if (
                _TLQ_WORKER_INSTANCE.solver is None
                or not _TLQ_WORKER_INSTANCE.solver.available(exception_flag=False)
            ):
                raise RuntimeError("TwoLevelQuadratic worker could not create an available solver.")

        if "combo_bb_threshold" in cfg:
            _TLQ_WORKER_INSTANCE.COMBO_BB_THRESHOLD = int(cfg["combo_bb_threshold"])

        # ---- keep Formulation 2 options synchronized on workers ----
        if "use_residual_correction" in cfg:
            _TLQ_WORKER_INSTANCE.use_residual_correction = bool(cfg["use_residual_correction"])
        if "geometry_threshold" in cfg:
            _TLQ_WORKER_INSTANCE.geometry_threshold = float(cfg["geometry_threshold"])
        if "rho_B" in cfg:
            _TLQ_WORKER_INSTANCE.rho_B = float(cfg["rho_B"])
        if "omega" in cfg:
            _TLQ_WORKER_INSTANCE.omega = float(cfg["omega"])
        if "residual_neighbors" in cfg:
            _TLQ_WORKER_INSTANCE.residual_neighbors = int(max(1, cfg["residual_neighbors"]))
        if "fill_distance_abs_tol" in cfg:
            _TLQ_WORKER_INSTANCE.fill_distance_abs_tol = float(cfg["fill_distance_abs_tol"])
        if "fill_distance_rel_tol" in cfg:
            _TLQ_WORKER_INSTANCE.fill_distance_rel_tol = float(cfg["fill_distance_rel_tol"])
        if "fill_distance_max_cells" in cfg:
            _TLQ_WORKER_INSTANCE.fill_distance_max_cells = int(max(1, cfg["fill_distance_max_cells"]))

        if cmd == "compute_region_models":
            tasks_by_rank = msg.get("tasks_by_rank", None)
            if not isinstance(tasks_by_rank, (list, tuple)):
                raise RuntimeError("TwoLevelQuadratic worker received malformed tasks_by_rank.")
            if rank >= len(tasks_by_rank):
                raise RuntimeError("TwoLevelQuadratic worker rank exceeds tasks_by_rank length.")

            local_tasks = tasks_by_rank[rank]

            local_payload = _TLQ_WORKER_INSTANCE._serial_region_models(
                tasks=local_tasks,
                bounds=np.asarray(cfg["bounds"], dtype=float),
                all_X=np.asarray(cfg["all_X"], dtype=float),
                all_Y_parts=np.asarray(cfg["all_Y_parts"], dtype=float),
                all_block_clouds=cfg["all_block_clouds"],
                specs=cfg["specs"],
            )

            comm.gather(local_payload, root=0)

        elif cmd == "compute_best_combo":
            payload = _TLQ_WORKER_INSTANCE._serial_best_combo_search_implicit(
                block_region_models=cfg["block_region_models"],
                d_master=int(cfg["d_master"]),
                rank=rank,
                size=size,
            )
            comm.gather(payload, root=0)

        elif cmd == "compute_best_combo_bb":
            payload = _TLQ_WORKER_INSTANCE._serial_best_combo_search_bb(
                block_region_models=cfg["block_region_models"],
                d_master=int(cfg["d_master"]),
                rank=rank,
                size=size,
            )
            comm.gather(payload, root=0)

        else:
            raise RuntimeError(f"TwoLevelQuadratic worker received unknown command '{cmd}'.")


def stop_two_level_workers():
    MPI = _parallel_mpi_if_active()
    if MPI is None:
        return
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0 or size <= 1:
        return

    comm.bcast({"cmd": "stop"}, root=0)


class TwoLevelQuadratic:
    """
    Generalized two-level quadratic underestimator with residual-corrected
    region lower models.

    The correction is computed independently for each block-partition (k,r):
        epsilon_{k,r} = (rho * Braw_{k,r} + omega) * h_{k,r}

    where Braw_{k,r} is estimated directly from sampled residuals
        e_i = f_k(z_i) - m_{k,r}(z_i).

    The correction is applied only as a constant shift:
        gamma_{k,r} <- gamma_{k,r} - epsilon_{k,r}

    Everything after region-model construction remains unchanged.
    """

    def __init__(self, intercept=True, solver="highs", sample_per_axis=5):
        self.intercept = bool(intercept)
        self.sample_per_axis = int(max(1, sample_per_axis))

        self.VERBOSE = False

        # If combo_count_after_pruning > COMBO_BB_THRESHOLD,
        # use branch-and-bound combo search instead of flat combo enumeration.
        self.COMBO_BB_THRESHOLD = 20

        # Parameters

        # ---- residual-corrected lower-bound options ----
        self.use_residual_correction = True
        self.geometry_threshold = float(GEOMETRY_THRESHOLD)
        self.omega = float(OMEGA_RESIDUAL)
        self.rho_B = float(RHO_RESIDUAL)
        self.residual_neighbors = int(max(1, RESIDUAL_NEIGHBORS))

        # IID sampling policy read by DDSBB.
        self.iid_target_multiplier = int(max(1, IID_TARGET_MULTIPLIER))
        self.iid_min_fresh_per_node = int(max(1, IID_MIN_FRESH_PER_NODE))
        self.iid_geometry_batch_multiplier = int(max(1, IID_GEOMETRY_BATCH_MULTIPLIER))
        self.max_geometry_enrich_samples = int(max(0, MAX_GEOMETRY_ENRICH_SAMPLES))

        # Certified fill-distance settings for coupling dimension > 1.
        self.fill_distance_abs_tol = float(max(0.0, FILL_DISTANCE_ABS_TOL))
        self.fill_distance_rel_tol = float(max(0.0, FILL_DISTANCE_REL_TOL))
        self.fill_distance_max_cells = int(max(1, FILL_DISTANCE_MAX_CELLS))
        self._fill_distance_cache = {}
        # F2 geometry-plan cache: neighbor selection/kappa/dN depend only on X.
        # Residual slopes Braw are deliberately recomputed from current residuals.
        self._f2_neighbor_plan_cache = {}

        self.solver_name = str(solver)
        self.solver = pe.SolverFactory(self.solver_name)
        if self.solver is None or not self.solver.available(exception_flag=False):
            self.solver_name = "highs"
            self.solver = pe.SolverFactory(self.solver_name)
        if self.solver is None or not self.solver.available(exception_flag=False):
            raise RuntimeError("TwoLevelQuadratic: neither requested solver nor 'highs' is available.")

        self.time_underestimate = 0.0

        self.last_parts = None
        self.last_total = None
        self.last_mpi = None
        self.last_timing_breakdown = None
        self.last_pruning_stats = None

        self.total_underestimator_fevals = 0
        self.block_underestimator_fevals = None
        self.total_combo_evals = 0

    def get_total_fevals(self):
        return int(self.total_underestimator_fevals)

    def get_block_fevals(self):
        if self.block_underestimator_fevals is None:
            return None
        return list(self.block_underestimator_fevals)

    @staticmethod
    def _minimize_separable_quadratic_on_unit_interval(a, b):
        a = float(a)
        b = float(b)

        if a > 1e-12:
            u = -b / (2.0 * a)
            u = min(1.0, max(0.0, u))
        else:
            if b > 0.0:
                u = 0.0
            elif b < 0.0:
                u = 1.0
            else:
                u = 0.5

        val = a * u * u + b * u
        return float(u), float(val)

    @staticmethod
    def _unique_rows_preserve_order(X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        _, idx = np.unique(X, axis=0, return_index=True)
        idx = np.sort(idx)
        return X[idx, :], Y[idx]

    @staticmethod
    def _alpha_from_directions(U, diag=None):
        """
        Compute
            alpha = min_{||v||_2=1} max_j |v^T u_j|
        from a finite set of unit directions U.

        Geometrically, alpha is the Euclidean inradius of conv(+/- U)
        centered at the origin.  For a full-dimensional direction set this
        is obtained from the facet distances of the symmetric convex hull.
        """
        _alpha_start = time.perf_counter() if diag is not None else None
        if diag is not None:
            diag["alpha_calls"] += 1
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[0] == 0:
            return 0.0

        d = int(U.shape[1])
        if d == 0:
            return 1.0

        norms = np.linalg.norm(U, axis=1)
        keep = norms > 1e-14
        U = U[keep, :]
        norms = norms[keep]
        if U.shape[0] == 0:
            return 0.0
        U = U / norms[:, None]

        _rank_start = time.perf_counter() if diag is not None else None
        _rank_value = np.linalg.matrix_rank(U, tol=1e-10)
        if diag is not None:
            diag["matrix_rank_calls"] += 1
            diag["matrix_rank_sec"] += time.perf_counter() - _rank_start
            diag["max_direction_rows"] = max(diag["max_direction_rows"], int(U.shape[0]))
        if _rank_value < d:
            if diag is not None:
                diag["rank_deficient_calls"] += 1
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return 0.0

        if d == 1:
            if diag is not None:
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return 1.0

        # General-purpose, dimension-aware computation. In 2D, all vertices
        # +/-U lie on the unit circle. Consecutive angular points form hull
        # edges; the edge distance from the origin is cos(angular_gap / 2).
        # Therefore the inradius is cos(maximum_angular_gap / 2). This is the
        # SAME geometric quantity as the Qhull facet-distance calculation,
        # without repeatedly creating a Qhull context on every MPI process.
        # No objective, benchmark, partition or sampling assumptions enter.
        if d == 2:
            angles = np.arctan2(U[:, 1], U[:, 0])
            angles = np.mod(np.concatenate((angles, angles + np.pi)), 2.0 * np.pi)
            angles.sort()
            gaps = np.diff(angles)
            wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
            largest_gap = max(float(np.max(gaps)), float(wrap_gap))
            alpha = float(min(1.0, max(0.0, np.cos(0.5 * largest_gap))))
            if diag is not None:
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return alpha

        # Dimension >= 3: retain the original, fully general Qhull path.
        P = np.vstack([U, -U])
        _hull_start = time.perf_counter() if diag is not None else None
        try:
            hull = ConvexHull(P)
        except QhullError:
            if diag is not None:
                diag["convex_hull_calls"] += 1
                diag["convex_hull_failures"] += 1
                diag["convex_hull_sec"] += time.perf_counter() - _hull_start
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return 0.0
        if diag is not None:
            diag["convex_hull_calls"] += 1
            diag["convex_hull_sec"] += time.perf_counter() - _hull_start
            diag["max_hull_points"] = max(diag["max_hull_points"], int(P.shape[0]))
            diag["max_hull_facets"] = max(diag["max_hull_facets"], int(len(hull.equations)))

        equations = np.asarray(hull.equations, dtype=float)
        normals = equations[:, :-1]
        offsets = equations[:, -1]
        normal_norms = np.linalg.norm(normals, axis=1)
        valid = normal_norms > 1e-14
        if not np.any(valid):
            if diag is not None:
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return 0.0

        distances = -offsets[valid] / normal_norms[valid]
        distances = distances[np.isfinite(distances)]
        distances = distances[distances > 0.0]
        if distances.size == 0:
            if diag is not None:
                diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
            return 0.0

        alpha = float(np.min(distances))
        if diag is not None:
            diag["alpha_total_sec"] += time.perf_counter() - _alpha_start
        return float(min(1.0, max(0.0, alpha)))

    def _residual_geometry_from_neighbors(self, X_train, residual_train, diag=None):
        """
        Build/reuse the residual-neighbor geometry plan.

        Neighbor selection, kappa, and dN depend only on the sample coordinates X
        and the geometry settings, so that work is cached.  Braw depends on the
        current residual values and is therefore recomputed on every call from
        the cached selected pairs.
        """
        _geom_start = time.perf_counter() if diag is not None else None
        X = np.asarray(X_train, dtype=float)
        R = np.asarray(residual_train, dtype=float).ravel()

        if X.ndim != 2:
            raise ValueError("TwoLevelQuadratic: X_train must be 2D for residual geometry.")
        if R.ndim != 1 or R.shape[0] != X.shape[0]:
            raise ValueError("TwoLevelQuadratic: residual_train mismatch.")

        X, R = self._unique_rows_preserve_order(X, R)
        X = np.ascontiguousarray(X, dtype=np.float64)
        n, d = X.shape
        if diag is not None:
            diag["geometry_calls"] += 1
            diag["training_points_sum"] += int(n)
            diag["max_training_points"] = max(diag["max_training_points"], int(n))
            diag["max_joint_dimension"] = max(diag["max_joint_dimension"], int(d))
        if n <= 1:
            return {"Braw": 0.0, "kappa": float("inf"), "dN": float("inf"), "neighbor_count_max": 0}

        # Exact coordinate/settings key.  The digest keeps the dictionary key
        # compact even when a regional training cloud is large.
        x_digest = hashlib.blake2b(X.view(np.uint8), digest_size=16).digest()
        cache_key = (
            tuple(X.shape), X.dtype.str, x_digest,
            float(self.geometry_threshold), int(self.residual_neighbors),
        )
        plan = self._f2_neighbor_plan_cache.get(cache_key)

        if plan is None:
            _dist_start = time.perf_counter() if diag is not None else None
            diff = X[:, None, :] - X[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            if diag is not None:
                diag["distance_matrix_sec"] += time.perf_counter() - _dist_start

            selected_plan = []
            kappa = 1.0
            dN = 0.0
            neighbor_count_max = 0
            alpha_target = 1.0 / float(self.geometry_threshold)

            for i in range(n):
                _sort_start = time.perf_counter() if diag is not None else None
                order = np.argsort(dist[i, :])
                order = order[np.isfinite(dist[i, order])]
                order = order[dist[i, order] > 1e-14]
                if diag is not None:
                    diag["neighbor_sort_sec"] += time.perf_counter() - _sort_start
                    diag["neighbor_points_processed"] += 1

                if order.size == 0:
                    return {"Braw": 0.0, "kappa": float("inf"), "dN": float("inf"), "neighbor_count_max": int(neighbor_count_max)}

                min_count = min(order.size, max(int(self.residual_neighbors), d))
                selected = order[:min_count]

                def _directions(indices):
                    dd_local = dist[i, indices]
                    return (X[indices, :] - X[i, :]) / dd_local[:, None]

                U = _directions(selected)
                alpha_i = self._alpha_from_directions(U, diag=diag)
                next_pos = min_count
                while next_pos < order.size and (alpha_i <= 0.0 or alpha_i + 1e-12 < alpha_target):
                    next_pos += 1
                    if diag is not None:
                        diag["neighbor_expansions"] += 1
                    selected = order[:next_pos]
                    U = _directions(selected)
                    alpha_i = self._alpha_from_directions(U, diag=diag)

                kappa_i = float("inf") if alpha_i <= 0.0 else 1.0 / float(alpha_i)
                dd = np.asarray(dist[i, selected], dtype=float).copy()
                selected = np.asarray(selected, dtype=int).copy()
                selected_plan.append((int(i), selected, dd))

                if dd.size > 0:
                    dN = max(dN, float(np.max(dd)))
                kappa = max(kappa, float(kappa_i))
                neighbor_count_max = max(neighbor_count_max, int(len(selected)))
                if diag is not None:
                    diag["max_selected_neighbors"] = max(diag["max_selected_neighbors"], int(len(selected)))

            plan = {
                "selected_plan": selected_plan,
                "kappa": float(kappa),
                "dN": float(dN),
                "neighbor_count_max": int(neighbor_count_max),
            }
            self._f2_neighbor_plan_cache[cache_key] = plan

        # Always recompute residual slopes: residuals change when the fitted
        # regional model changes, even if the coordinate geometry is identical.
        Braw = 0.0
        for i, selected, dd in plan["selected_plan"]:
            dy = np.abs(R[int(i)] - R[selected])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.divide(dy, dd)
            ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
            if ratios.size > 0:
                Braw = max(Braw, float(np.max(ratios)))

        if diag is not None:
            diag["geometry_total_sec"] += time.perf_counter() - _geom_start
        return {
            "Braw": float(Braw),
            "kappa": float(plan["kappa"]),
            "dN": float(plan["dN"]),
            "neighbor_count_max": int(plan["neighbor_count_max"]),
        }

    @staticmethod
    def _fill_distance_1d_exact(points):
        """
        Exact fill distance on [0,1] for scattered 1D points:
            max{s_1 - 0, 1 - s_N, max_i (s_{i+1} - s_i)/2}.
        """
        pts = np.asarray(points, dtype=float).ravel()
        pts = pts[np.isfinite(pts)]
        pts = pts[(pts >= 0.0) & (pts <= 1.0)]
        pts = np.sort(np.unique(pts))

        if pts.size == 0:
            return 1.0

        left_gap = float(pts[0])
        right_gap = float(1.0 - pts[-1])
        interior_gap = 0.0
        if pts.size >= 2:
            interior_gap = float(0.5 * np.max(np.diff(pts)))

        return float(max(left_gap, right_gap, interior_gap))

    @staticmethod
    def _nearest_sample_distance(point, samples):
        point = np.asarray(point, dtype=float).reshape(1, -1)
        samples = np.asarray(samples, dtype=float)
        return float(np.min(np.linalg.norm(samples - point, axis=1)))

    def _fill_distance_nd_certified(self, points):
        """
        Certified bounds L <= h_S <= U on [0,1]^d, d > 1.

        For each axis-aligned cell C with center c and circumradius r_C,
        g(s)=min_i ||s-s_i|| is 1-Lipschitz, hence

            g(c) <= h_S,
            sup_{s in C} g(s) <= g(c) + r_C.

        The cell having the largest current upper bound is bisected along its
        longest side until the global absolute or relative gap tolerance is
        met, or the cell cap is reached.  No black-box evaluations occur.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("TwoLevelQuadratic: master points must be 2D for certified fill distance.")

        n, d = pts.shape
        if d <= 1:
            raise ValueError("TwoLevelQuadratic: certified ND fill distance requires d > 1.")
        if n == 0:
            val = float(np.sqrt(d))
            return val, val, 1

        pts = np.unique(np.clip(pts, 0.0, 1.0), axis=0)

        abs_tol = float(self.fill_distance_abs_tol)
        rel_tol = float(self.fill_distance_rel_tol)
        max_cells = int(max(1, self.fill_distance_max_cells))

        counter = 0
        heap = []
        lower_global = 0.0

        def make_cell(lo, hi):
            nonlocal counter, lower_global
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            center = 0.5 * (lo + hi)
            radius = 0.5 * float(np.linalg.norm(hi - lo))
            g_center = self._nearest_sample_distance(center, pts)
            lower_global = max(lower_global, g_center)
            upper = min(float(np.sqrt(d)), g_center + radius)
            counter += 1
            return (-upper, counter, lo, hi, g_center)

        heapq.heappush(heap, make_cell(np.zeros(d), np.ones(d)))
        cells_created = 1

        while heap:
            upper_global = max(lower_global, -float(heap[0][0]))
            gap = max(0.0, upper_global - lower_global)
            rel_gap = gap / max(upper_global, 1e-15)

            if gap <= abs_tol or rel_gap <= rel_tol:
                break
            if cells_created >= max_cells:
                break

            _, _, lo, hi, _ = heapq.heappop(heap)
            widths = hi - lo
            axis = int(np.argmax(widths))
            mid = 0.5 * (lo[axis] + hi[axis])

            lo1 = lo.copy()
            hi1 = hi.copy()
            hi1[axis] = mid

            lo2 = lo.copy()
            hi2 = hi.copy()
            lo2[axis] = mid

            heapq.heappush(heap, make_cell(lo1, hi1))
            cells_created += 1
            if cells_created >= max_cells:
                break
            heapq.heappush(heap, make_cell(lo2, hi2))
            cells_created += 1

        upper_global = lower_global
        if heap:
            upper_global = max(lower_global, -float(heap[0][0]))

        return float(lower_global), float(upper_global), int(cells_created)

    def _estimate_fill_distance_kr(self, X_train, n_private, sample_per_axis):
        """
        Return certified lower/upper bounds for h_{k,r} in the local fitting
        coordinates z=[private-local, local-master] in [0,1]^d.

        The private-grid fill distance is known exactly.  The coupling-space
        fill distance is exact in one dimension and certified to a tolerance
        in multiple dimensions.
        """
        X = np.asarray(X_train, dtype=float)
        if X.ndim != 2:
            raise ValueError("TwoLevelQuadratic: X_train must be 2D for fill-distance estimate.")

        private_dim = int(n_private)
        total_dim = int(X.shape[1])
        master_dim = int(total_dim - private_dim)

        if private_dim < 0 or master_dim < 0:
            raise ValueError("TwoLevelQuadratic: invalid private/master dimension split for h_{k,r}.")

        m = int(max(1, sample_per_axis))
        if private_dim == 0:
            h_private = 0.0
        elif m <= 1:
            h_private = 0.5 * float(np.sqrt(private_dim))
        else:
            h_private = float(np.sqrt(private_dim) / (2.0 * (m - 1)))

        if master_dim == 0:
            h_master_lower = 0.0
            h_master_upper = 0.0
            cells_used = 0
        else:
            S = np.asarray(X[:, private_dim:], dtype=float)
            S = np.unique(np.clip(S, 0.0, 1.0), axis=0)

            if master_dim == 1:
                h_exact = self._fill_distance_1d_exact(S[:, 0])
                h_master_lower = h_exact
                h_master_upper = h_exact
                cells_used = 0
            else:
                key = (
                    int(master_dim),
                    tuple(np.round(S, 14).ravel().tolist()),
                    tuple(S.shape),
                    float(self.fill_distance_abs_tol),
                    float(self.fill_distance_rel_tol),
                    int(self.fill_distance_max_cells),
                )
                cached = self._fill_distance_cache.get(key, None)
                if cached is None:
                    cached = self._fill_distance_nd_certified(S)
                    self._fill_distance_cache[key] = cached
                h_master_lower, h_master_upper, cells_used = cached

        h_lower = float(np.sqrt(h_private ** 2 + h_master_lower ** 2))
        h_upper = float(np.sqrt(h_private ** 2 + h_master_upper ** 2))

        return {
            "h_lower": h_lower,
            "h_upper": h_upper,
            "h_private": float(h_private),
            "h_master_lower": float(h_master_lower),
            "h_master_upper": float(h_master_upper),
            "fill_cells": int(cells_used),
        }

    def _fit_region_ddcu_nd(self, X_train, Y_train):
        """
        Fit a separable convex quadratic underestimator on local variables z in [0,1]^d:
            ell(z) = sum_j a_j z_j^2 + b_j z_j + c

        Original sampled DDCU/TLQ LP:
            min sum_i (Y_i - ell(z_i))
            s.t. ell(z_i) <= Y_i
                 a_j >= 0

        No objective regularization, no model-slope variable, and no Lipschitz
        derivative constraints are used in this residual-corrected version.

        Returns:
            a, b, c
        """
        X_train = np.asarray(X_train, dtype=float)
        Y_train = np.asarray(Y_train, dtype=float).ravel()

        if X_train.ndim != 2:
            raise ValueError("TwoLevelQuadratic: X_train must be 2D.")
        if Y_train.ndim != 1:
            raise ValueError("TwoLevelQuadratic: Y_train must be 1D.")
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError("TwoLevelQuadratic: X_train and Y_train row mismatch.")
        if X_train.shape[0] == 0:
            raise ValueError("TwoLevelQuadratic: cannot fit region model with zero samples.")

        X_train, Y_train = self._unique_rows_preserve_order(X_train, Y_train)

        n_samp = int(X_train.shape[0])
        dim = int(X_train.shape[1])

        _profile_fit_start = time.perf_counter()
        m = pe.ConcreteModel()
        m.I = pe.RangeSet(0, n_samp - 1)
        m.J = pe.RangeSet(0, dim - 1)

        m.a = pe.Var(m.J, within=pe.NonNegativeReals, initialize=0.0)
        m.b = pe.Var(m.J, within=pe.Reals, initialize=0.0)

        if self.intercept:
            m.c = pe.Var(within=pe.Reals, initialize=float(np.min(Y_train)))
        else:
            corner = np.where(np.all(np.isclose(X_train, 0.0), axis=1))[0]
            c_fixed = float(np.min(Y_train[corner])) if len(corner) > 0 else float(np.min(Y_train))
            m.c = pe.Param(initialize=c_fixed, mutable=False)

        def ell_expr(mm, i):
            i = int(i)
            return sum(
                mm.a[j] * float(X_train[i, j] ** 2) + mm.b[j] * float(X_train[i, j])
                for j in range(dim)
            ) + mm.c

        def under_rule(mm, i):
            return ell_expr(mm, i) <= float(Y_train[int(i)])

        m.under = pe.Constraint(m.I, rule=under_rule)

        def obj_rule(mm):
            return sum(float(Y_train[i]) - ell_expr(mm, i) for i in range(n_samp))

        m.obj = pe.Objective(rule=obj_rule, sense=pe.minimize)

        _profile_solve_start = time.perf_counter()
        result = self.solver.solve(m)
        _profile_solve_end = time.perf_counter()
        self._last_fit_profile = {
            "pyomo_build_sec": _profile_solve_start - _profile_fit_start,
            "lp_solve_sec": _profile_solve_end - _profile_solve_start,
        }

        status = getattr(result.solver, "status", None)
        term = getattr(result.solver, "termination_condition", None)
        bad_terms = {
            pe.TerminationCondition.infeasible,
            pe.TerminationCondition.unbounded,
            pe.TerminationCondition.infeasibleOrUnbounded,
            pe.TerminationCondition.error,
            pe.TerminationCondition.invalidProblem,
        }

        if status not in (pe.SolverStatus.ok, pe.SolverStatus.warning):
            raise RuntimeError(f"TwoLevelQuadratic: solver failed with status={status}, termination={term}.")
        if term in bad_terms:
            raise RuntimeError(f"TwoLevelQuadratic: solver terminated badly with termination={term}.")

        a = np.array([pe.value(m.a[j]) for j in range(dim)], dtype=float)
        b = np.array([pe.value(m.b[j]) for j in range(dim)], dtype=float)
        c = float(pe.value(m.c))

        tol = 1e-5
        if np.any(a < -tol):
            raise ValueError("TwoLevelQuadratic: negative quadratic coefficient detected.")

        a[np.abs(a) < 1e-14] = 0.0
        b[np.abs(b) < 1e-14] = 0.0

        return a, b, c


    @staticmethod
    def _eval_separable_quadratic(X, a, b, c):
        """
        Evaluate ell(z) = sum_j a_j z_j^2 + b_j z_j + c at rows of X.
        """
        X = np.asarray(X, dtype=float)
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("TwoLevelQuadratic: X must be 2D for quadratic evaluation.")
        if X.shape[1] != len(a) or X.shape[1] != len(b):
            raise ValueError("TwoLevelQuadratic: coefficient length mismatch in quadratic evaluation.")
        return np.sum(a[None, :] * X * X + b[None, :] * X, axis=1) + float(c)


    def _normalize_block_specs(self, *, block_specs):
        """Validate and normalize the canonical DBDDSBB block specification."""
        if block_specs is None:
            raise ValueError("TwoLevelQuadratic: missing block_specs.")

        out = []
        for k, spec in enumerate(block_specs):
            required = [
                "name",
                "master_indices",
                "private_names",
                "private_partitions",
                "block_function",
                "assemble_args",
            ]
            missing = [key for key in required if key not in spec]
            if missing:
                raise ValueError(f"TwoLevelQuadratic: block_specs[{k}] missing keys {missing}.")

            master_indices = [int(j) for j in spec["master_indices"]]
            private_names = [str(v) for v in spec["private_names"]]
            private_partitions_k = spec["private_partitions"]

            if len(private_partitions_k) != len(private_names):
                raise ValueError(
                    f"TwoLevelQuadratic: block_specs[{k}] has "
                    f"{len(private_names)} private_names but "
                    f"{len(private_partitions_k)} private_partitions axis-lists."
                )

            out.append({
                "name": str(spec["name"]),
                "master_indices": master_indices,
                "private_names": private_names,
                "private_partitions": private_partitions_k,
                "block_function": spec["block_function"],
                "assemble_args": spec["assemble_args"],
            })

        return out

    @staticmethod
    def _validate_interval_list(interval_list, msg_prefix):
        out = []
        for p in interval_list:
            if len(p) != 2:
                raise ValueError(f"{msg_prefix}: each interval must be a 2-tuple (lb, ub).")
            lbp = float(p[0])
            ubp = float(p[1])
            if lbp > ubp:
                raise ValueError(f"{msg_prefix}: interval lower bound exceeds upper bound.")
            out.append((lbp, ubp))
        if len(out) == 0:
            raise ValueError(f"{msg_prefix}: interval list cannot be empty.")
        return out

    def _get_private_boxes_for_block(self, spec):
        axis_lists = []
        for j, parts_j in enumerate(spec["private_partitions"]):
            axis_lists.append(
                self._validate_interval_list(
                    parts_j,
                    msg_prefix=f"TwoLevelQuadratic: block '{spec['name']}' axis {j}",
                )
            )

        boxes = list(itertools.product(*axis_lists))
        if len(boxes) == 0:
            raise ValueError(f"TwoLevelQuadratic: block '{spec['name']}' has no private boxes.")
        return boxes

    @staticmethod
    def _point_in_box(point, private_box, tol=1e-12):
        p = np.asarray(point, dtype=float).ravel()
        if len(p) != len(private_box):
            return False
        for j, (lbj, ubj) in enumerate(private_box):
            if p[j] < lbj - tol or p[j] > ubj + tol:
                return False
        return True

    def _build_region_training_data_from_stored_cloud(
        self,
        *,
        spec,
        block_index,
        private_box,
        bounds,
        all_X,
        all_block_clouds,
    ):
        bounds = np.asarray(bounds, dtype=float)
        all_X = np.asarray(all_X, dtype=float)

        master_indices = list(spec["master_indices"])
        n_private = len(spec["private_names"])

        if len(private_box) != n_private:
            raise ValueError("TwoLevelQuadratic: private_box length mismatch.")
        if len(all_block_clouds) != all_X.shape[0]:
            raise ValueError("TwoLevelQuadratic: all_block_clouds row mismatch with all_X.")

        X_rows = []
        Y_rows = []

        for i in range(all_X.shape[0]):
            x_scaled_full = np.asarray(all_X[i, :], dtype=float)
            local_master_scaled = x_scaled_full[master_indices]

            cloud_i = all_block_clouds[i]
            if block_index < 0 or block_index >= len(cloud_i):
                raise ValueError("TwoLevelQuadratic: invalid block_index for stored cloud.")

            block_cloud = cloud_i[block_index]
            private_points = np.asarray(block_cloud["private_points"], dtype=float)
            block_values = np.asarray(block_cloud["block_values"], dtype=float).ravel()

            if private_points.ndim != 2:
                raise ValueError("TwoLevelQuadratic: stored private_points must be 2D.")
            if private_points.shape[0] != len(block_values):
                raise ValueError("TwoLevelQuadratic: stored private_points/block_values mismatch.")

            for pvals, yval in zip(private_points, block_values):
                if not self._point_in_box(pvals, private_box):
                    continue

                private_local = []
                for j in range(n_private):
                    lbj, ubj = private_box[j]
                    if abs(ubj - lbj) <= 1e-15:
                        private_local.append(0.5)
                    else:
                        private_local.append((pvals[j] - lbj) / (ubj - lbj))

                row = np.concatenate([
                    np.asarray(private_local, dtype=float),
                    np.asarray(local_master_scaled, dtype=float),
                ])
                X_rows.append(row)
                Y_rows.append(float(yval))

        if len(X_rows) == 0:
            raise ValueError(
                f"TwoLevelQuadratic: no stored sampled points found in region for block '{spec['name']}'."
            )

        X_train = np.asarray(X_rows, dtype=float)
        Y_train = np.asarray(Y_rows, dtype=float)
        return X_train, Y_train

    def _region_reduced_quadratic(self, a, b, c, n_private, master_indices, d_master):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        n_master_local = len(master_indices)
        if len(a) != n_private + n_master_local:
            raise ValueError("TwoLevelQuadratic: coefficient length mismatch in region reduction.")

        gamma = float(c)

        for j in range(n_private):
            _, vj = self._minimize_separable_quadratic_on_unit_interval(a[j], b[j])
            gamma += float(vj)

        alpha_global = np.zeros(d_master, dtype=float)
        beta_global = np.zeros(d_master, dtype=float)

        for ell, j_global in enumerate(master_indices):
            idx = n_private + ell
            alpha_global[j_global] = float(a[idx])
            beta_global[j_global] = float(b[idx])

        return {
            "alpha_global": alpha_global,
            "beta_global": beta_global,
            "gamma": float(gamma),
        }

    @staticmethod
    def _minimize_sum_of_region_quadratics_over_unit_box(alpha_global, beta_global, gamma):
        alpha_global = np.asarray(alpha_global, dtype=float)
        beta_global = np.asarray(beta_global, dtype=float)

        d = len(alpha_global)
        t_star = np.zeros(d, dtype=float)
        val = float(gamma)

        for j in range(d):
            tj, vj = TwoLevelQuadratic._minimize_separable_quadratic_on_unit_interval(
                alpha_global[j], beta_global[j]
            )
            t_star[j] = float(tj)
            val += float(vj)

        return np.asarray(t_star, dtype=float), float(val)

    @staticmethod
    def _split_tasks_static(tasks, size):
        chunks = [[] for _ in range(size)]
        for i, task in enumerate(tasks):
            chunks[i % size].append(task)
        return chunks

    def _serial_region_models(self, tasks, bounds, all_X, all_Y_parts, all_block_clouds, specs):
        local_results = []
        n_blocks = len(specs)

        local_total_fevals = 0
        local_block_fevals = [0] * n_blocks
        _profile_keys = ("training_sec", "fill_distance_sec", "pyomo_build_sec",
                         "lp_solve_sec", "fit_other_sec", "residual_geometry_sec",
                         "other_sec", "task_total_sec")
        _profile_totals = {key: 0.0 for key in _profile_keys}
        _profile_max_task = (0.0, None)
        MPI = _parallel_mpi_if_active()
        _profile_rank = int(MPI.COMM_WORLD.Get_rank()) if MPI is not None else 0
        _geometry_keys = (
            "geometry_total_sec", "distance_matrix_sec", "neighbor_sort_sec",
            "alpha_total_sec", "matrix_rank_sec", "convex_hull_sec",
            "geometry_calls", "training_points_sum", "max_training_points",
            "max_joint_dimension", "neighbor_points_processed", "neighbor_expansions",
            "alpha_calls", "matrix_rank_calls", "rank_deficient_calls",
            "convex_hull_calls", "convex_hull_failures", "max_direction_rows",
            "max_hull_points", "max_hull_facets", "max_selected_neighbors",
        )
        _geometry_diag = {key: 0 for key in _geometry_keys}

        for task in tasks:
            _task_start = time.perf_counter()

            k = int(task["block_index"])
            r = int(task["region_index"])
            private_box = tuple(task["private_box"])
            spec = specs[k]

            X_train, Y_train = self._build_region_training_data_from_stored_cloud(
                spec=spec,
                block_index=k,
                private_box=private_box,
                bounds=bounds,
                all_X=all_X,
                all_block_clouds=all_block_clouds,
            )

            _t_training = time.perf_counter()
            _profile_totals["training_sec"] += _t_training - _task_start
            n_private = len(spec["private_names"])

            if self.use_residual_correction:
                h_info = self._estimate_fill_distance_kr(
                    X_train=X_train,
                    n_private=n_private,
                    sample_per_axis=self.sample_per_axis,
                )
                h_lower_kr = float(h_info["h_lower"])
                h_kr = float(h_info["h_upper"])
            else:
                h_info = {
                    "h_lower": 0.0,
                    "h_upper": 0.0,
                    "h_private": 0.0,
                    "h_master_lower": 0.0,
                    "h_master_upper": 0.0,
                    "fill_cells": 0,
                }
                h_lower_kr = 0.0
                h_kr = 0.0

            _t_fill = time.perf_counter()
            _profile_totals["fill_distance_sec"] += _t_fill - _t_training
            _t_fit = time.perf_counter()
            a, b, c = self._fit_region_ddcu_nd(
                X_train,
                Y_train,
            )

            _t_fit_end = time.perf_counter()
            _fit_profile = self._last_fit_profile
            _profile_totals["pyomo_build_sec"] += _fit_profile["pyomo_build_sec"]
            _profile_totals["lp_solve_sec"] += _fit_profile["lp_solve_sec"]
            _profile_totals["fit_other_sec"] += max(
                0.0, (_t_fit_end - _t_fit)
                - _fit_profile["pyomo_build_sec"] - _fit_profile["lp_solve_sec"]
            )
            _t_geometry = time.perf_counter()
            if self.use_residual_correction:
                m_train = self._eval_separable_quadratic(X_train, a, b, c)
                residual_train = np.asarray(Y_train, dtype=float).ravel() - np.asarray(m_train, dtype=float).ravel()

                geom = self._residual_geometry_from_neighbors(
                    X_train=X_train,
                    residual_train=residual_train,
                    diag=_geometry_diag,
                )
                Braw_kr = float(geom["Braw"])
                kappa_kr = float(geom["kappa"])
                dN_kr = float(geom["dN"])
                neighbor_count_kr = int(geom["neighbor_count_max"])

                # rho is supplied independently. The additive omega safeguard removes the
                # Braw=0 degeneracy while the whole correction still vanishes
                # with h under refinement.
                Bhat_kr = float(self.rho_B) * Braw_kr + float(self.omega)
                epsilon_kr = Bhat_kr * float(h_kr)

                directional_ok = bool(kappa_kr <= self.geometry_threshold + 1e-12)
                locality_ok = bool(dN_kr <= self.geometry_threshold * h_lower_kr + 1e-12)
                geometry_ok = bool(directional_ok and locality_ok)
            else:
                Braw_kr = 0.0
                Bhat_kr = 0.0
                epsilon_kr = 0.0
                kappa_kr = 1.0
                dN_kr = 0.0
                neighbor_count_kr = 0
                directional_ok = True
                locality_ok = True
                geometry_ok = True

            _t_geometry_end = time.perf_counter()
            _profile_totals["residual_geometry_sec"] += _t_geometry_end - _t_geometry
            reduced = self._region_reduced_quadratic(
                a=a,
                b=b,
                c=c,
                n_private=n_private,
                master_indices=spec["master_indices"],
                d_master=all_X.shape[1],
            )

            if self.use_residual_correction:
                reduced["gamma"] = float(reduced["gamma"] - epsilon_kr)

            local_results.append({
                "block_index": k,
                "region_index": r,
                "block_name": str(spec["name"]),
                "private_box": private_box,
                "a_fit": np.asarray(a, dtype=float).copy(),
                "b_fit": np.asarray(b, dtype=float).copy(),
                "c_fit": float(c),
                "alpha_global": np.asarray(reduced["alpha_global"], dtype=float).copy(),
                "beta_global": np.asarray(reduced["beta_global"], dtype=float).copy(),
                "gamma": float(reduced["gamma"]),
                "epsilon_kr": float(epsilon_kr),
                "Braw_kr": float(Braw_kr),
                "Bhat_kr": float(Bhat_kr),
                "h_kr": float(h_kr),
                "h_lower_kr": float(h_lower_kr),
                "h_private": float(h_info["h_private"]),
                "h_master_lower": float(h_info["h_master_lower"]),
                "h_master_upper": float(h_info["h_master_upper"]),
                "fill_cells": int(h_info["fill_cells"]),
                "kappa_kr": float(kappa_kr),
                "dN_kr": float(dN_kr),
                "neighbor_count_kr": int(neighbor_count_kr),
                "directional_ok": bool(directional_ok),
                "locality_ok": bool(locality_ok),
                "geometry_ok": bool(geometry_ok),
                "Gamma0": float(self.geometry_threshold),
                "omega": float(self.omega),
                "rho_B": float(self.rho_B),
            })

            _task_end = time.perf_counter()
            _profile_totals["other_sec"] += _task_end - _t_geometry_end
            _task_elapsed = _task_end - _task_start
            _profile_totals["task_total_sec"] += _task_elapsed
            if _task_elapsed > _profile_max_task[0]:
                _profile_max_task = (_task_elapsed, (k, r))

        return {
            "_diagnostic_profile": {
                "rank": _profile_rank,
                "task_count": len(tasks),
                "totals": _profile_totals,
                "max_task_sec": _profile_max_task[0],
                "max_task_id": _profile_max_task[1],
                "geometry_detail": _geometry_diag,
            },
            "results": local_results,
            "total_fevals": int(local_total_fevals),
            "block_fevals": list(local_block_fevals),
        }

    @staticmethod
    def _flatten_gathered(gathered_results):
        out = []
        total_fevals = 0
        block_fevals = None

        for chunk in gathered_results:
            if chunk is None:
                continue

            out.extend(chunk["results"])
            total_fevals += int(chunk["total_fevals"])

            if block_fevals is None:
                block_fevals = [0] * len(chunk["block_fevals"])

            for k, v in enumerate(chunk["block_fevals"]):
                block_fevals[k] += int(v)

        if block_fevals is None:
            block_fevals = []

        return out, total_fevals, block_fevals

    @staticmethod
    def _assemble_block_region_models(all_region_models, n_blocks):
        block_region_models = [[] for _ in range(n_blocks)]
        for rec in all_region_models:
            block_region_models[int(rec["block_index"])].append(rec)

        for k in range(n_blocks):
            block_region_models[k].sort(key=lambda z: int(z["region_index"]))

        return block_region_models

    def _parallel_region_models_controller_workers(self, tasks, bounds, all_X, all_Y_parts, all_block_clouds, specs):
        MPI = _parallel_mpi_if_active()
        if MPI is None:
            raise RuntimeError("TwoLevelQuadratic: MPI is not active for controller/worker mode.")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank != 0:
            raise RuntimeError(
                "TwoLevelQuadratic: only rank 0 may call _underestimate(). "
                "Non-root ranks must be in run_two_level_worker_loop()."
            )

        tasks_by_rank = self._split_tasks_static(tasks, size)

        msg = {
            "cmd": "compute_region_models",
            "config": {
                "intercept": bool(self.intercept),
                "solver_name": str(self.solver_name),
                "sample_per_axis": int(self.sample_per_axis),
                "combo_bb_threshold": int(self.COMBO_BB_THRESHOLD),
                "use_residual_correction": bool(self.use_residual_correction),
                "geometry_threshold": float(self.geometry_threshold),
                "rho_B": float(self.rho_B),
                "omega": float(self.omega),
                "residual_neighbors": int(self.residual_neighbors),
                "fill_distance_abs_tol": float(self.fill_distance_abs_tol),
                "fill_distance_rel_tol": float(self.fill_distance_rel_tol),
                "fill_distance_max_cells": int(self.fill_distance_max_cells),
                "bounds": np.asarray(bounds, dtype=float),
                "all_X": np.asarray(all_X, dtype=float),
                "all_Y_parts": np.asarray(all_Y_parts, dtype=float),
                "all_block_clouds": all_block_clouds,
                "specs": specs,
            },
            "tasks_by_rank": tasks_by_rank,
        }

        comm.bcast(msg, root=0)

        local_payload = self._serial_region_models(
            tasks=tasks_by_rank[0],
            bounds=bounds,
            all_X=all_X,
            all_Y_parts=all_Y_parts,
            all_block_clouds=all_block_clouds,
            specs=specs,
        )

        gathered_payloads = comm.gather(local_payload, root=0)
        return gathered_payloads

    def _serial_best_combo_search_implicit(self, block_region_models, d_master, rank, size):
        region_index_ranges = [range(len(models_k)) for models_k in block_region_models]

        best_val = INFINITY
        best_t = np.full(d_master, 0.5, dtype=float)
        best_combo = None
        best_alpha = None
        best_beta = None
        best_gamma = None
        combo_count = 0

        for combo_id, combo in enumerate(itertools.product(*region_index_ranges)):
            if combo_id % size != rank:
                continue

            combo_count += 1

            alpha = np.zeros(d_master, dtype=float)
            beta = np.zeros(d_master, dtype=float)
            gamma = 0.0

            for k, r in enumerate(combo):
                rec = block_region_models[k][r]
                alpha += np.asarray(rec["alpha_global"], dtype=float)
                beta += np.asarray(rec["beta_global"], dtype=float)
                gamma += float(rec["gamma"])

            t_star, val = self._minimize_sum_of_region_quadratics_over_unit_box(
                alpha_global=alpha,
                beta_global=beta,
                gamma=gamma,
            )

            if val < best_val:
                best_val = float(val)
                best_t = np.asarray(t_star, dtype=float).copy()
                best_combo = tuple(int(v) for v in combo)
                best_alpha = alpha.copy()
                best_beta = beta.copy()
                best_gamma = float(gamma)

        return {
            "best_val": float(best_val),
            "best_t": np.asarray(best_t, dtype=float).copy(),
            "best_combo": best_combo,
            "best_alpha": None if best_alpha is None else np.asarray(best_alpha, dtype=float).copy(),
            "best_beta": None if best_beta is None else np.asarray(best_beta, dtype=float).copy(),
            "best_gamma": None if best_gamma is None else float(best_gamma),
            "combo_count": int(combo_count),
            "bb_node_count": 0,
            "bb_pruned_count": 0,
            "bb_full_combo_count": int(combo_count),
        }

    @staticmethod
    def _select_global_best_combo(gathered_payloads, d_master):
        best_val = INFINITY
        best_t = np.full(d_master, 0.5, dtype=float)
        best_combo = None
        best_alpha = None
        best_beta = None
        best_gamma = None
        total_combo_count = 0
        total_bb_node_count = 0
        total_bb_pruned_count = 0
        total_bb_full_combo_count = 0

        for payload in gathered_payloads:
            if payload is None:
                continue

            total_combo_count += int(payload.get("combo_count", 0))
            total_bb_node_count += int(payload.get("bb_node_count", 0))
            total_bb_pruned_count += int(payload.get("bb_pruned_count", 0))
            total_bb_full_combo_count += int(payload.get("bb_full_combo_count", payload.get("combo_count", 0)))

            val = float(payload["best_val"])

            if val < best_val:
                best_val = val
                best_t = np.asarray(payload["best_t"], dtype=float).copy()
                best_combo = payload["best_combo"]
                best_alpha = None if payload["best_alpha"] is None else np.asarray(payload["best_alpha"], dtype=float).copy()
                best_beta = None if payload["best_beta"] is None else np.asarray(payload["best_beta"], dtype=float).copy()
                best_gamma = None if payload["best_gamma"] is None else float(payload["best_gamma"])

        return {
            "best_val": float(best_val),
            "best_t": np.asarray(best_t, dtype=float).copy(),
            "best_combo": best_combo,
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "best_gamma": best_gamma,
            "total_combo_count": int(total_combo_count),
            "total_bb_node_count": int(total_bb_node_count),
            "total_bb_pruned_count": int(total_bb_pruned_count),
            "total_bb_full_combo_count": int(total_bb_full_combo_count),
        }

    def _parallel_best_combo_controller_workers(self, block_region_models, d_master):
        MPI = _parallel_mpi_if_active()
        if MPI is None:
            raise RuntimeError("TwoLevelQuadratic: MPI is not active for controller/worker mode.")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank != 0:
            raise RuntimeError(
                "TwoLevelQuadratic: only rank 0 may call _underestimate(). "
                "Non-root ranks must be in run_two_level_worker_loop()."
            )

        msg = {
            "cmd": "compute_best_combo",
            "config": {
                "intercept": bool(self.intercept),
                "solver_name": str(self.solver_name),
                "sample_per_axis": int(self.sample_per_axis),
                "combo_bb_threshold": int(self.COMBO_BB_THRESHOLD),
                "block_region_models": block_region_models,
                "d_master": int(d_master),
            },
        }

        comm.bcast(msg, root=0)

        local_payload = self._serial_best_combo_search_implicit(
            block_region_models=block_region_models,
            d_master=int(d_master),
            rank=0,
            size=size,
        )

        gathered_payloads = comm.gather(local_payload, root=0)
        return gathered_payloads

    def _parallel_best_combo_bb_controller_workers(self, block_region_models, d_master):
        MPI = _parallel_mpi_if_active()
        if MPI is None:
            raise RuntimeError("TwoLevelQuadratic: MPI is not active for controller/worker mode.")

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank != 0:
            raise RuntimeError(
                "TwoLevelQuadratic: only rank 0 may call _underestimate(). "
                "Non-root ranks must be in run_two_level_worker_loop()."
            )

        msg = {
            "cmd": "compute_best_combo_bb",
            "config": {
                "intercept": bool(self.intercept),
                "solver_name": str(self.solver_name),
                "sample_per_axis": int(self.sample_per_axis),
                "combo_bb_threshold": int(self.COMBO_BB_THRESHOLD),
                "block_region_models": block_region_models,
                "d_master": int(d_master),
            },
        }

        comm.bcast(msg, root=0)

        local_payload = self._serial_best_combo_search_bb(
            block_region_models=block_region_models,
            d_master=int(d_master),
            rank=0,
            size=size,
        )

        gathered_payloads = comm.gather(local_payload, root=0)
        return gathered_payloads

    @staticmethod
    def _combo_count_from_block_region_models(block_region_models):
        count = 1
        for models_k in block_region_models:
            count *= int(len(models_k))
        return int(count)

    @staticmethod
    def _min_difference_between_region_models(rec_i, rec_j):
        alpha_i = np.asarray(rec_i["alpha_global"], dtype=float)
        beta_i = np.asarray(rec_i["beta_global"], dtype=float)
        gamma_i = float(rec_i["gamma"])

        alpha_j = np.asarray(rec_j["alpha_global"], dtype=float)
        beta_j = np.asarray(rec_j["beta_global"], dtype=float)
        gamma_j = float(rec_j["gamma"])

        da = alpha_i - alpha_j
        db = beta_i - beta_j
        dc = gamma_i - gamma_j

        _, min_val = TwoLevelQuadratic._minimize_sum_of_region_quadratics_over_unit_box(
            alpha_global=da,
            beta_global=db,
            gamma=dc,
        )

        return float(min_val)

    def _prune_dominated_regions_one_block(self, models_k, tol=1e-9):
        R = int(len(models_k))

        stats = {
            "before": R,
            "after": R,
            "removed": 0,
            "pair_tests": 0,
            "dominance_hits": 0,
        }

        if R <= 1:
            return list(models_k), stats

        dominated = [False] * R

        for i in range(R):
            for j in range(R):
                if i == j:
                    continue

                stats["pair_tests"] += 1

                min_ij = self._min_difference_between_region_models(models_k[i], models_k[j])

                if min_ij >= -float(tol):
                    min_ji = self._min_difference_between_region_models(models_k[j], models_k[i])

                    if min_ji >= -float(tol):
                        idx_i = int(models_k[i].get("region_index", i))
                        idx_j = int(models_k[j].get("region_index", j))

                        if idx_i > idx_j:
                            dominated[i] = True
                            stats["dominance_hits"] += 1
                            break
                        else:
                            continue

                    dominated[i] = True
                    stats["dominance_hits"] += 1
                    break

        pruned = [models_k[i] for i in range(R) if not dominated[i]]

        if len(pruned) == 0:
            pruned = [models_k[0]]

        stats["after"] = int(len(pruned))
        stats["removed"] = int(R - len(pruned))

        return pruned, stats

    def _prune_dominated_regions_by_block(self, block_region_models, tol=1e-9):
        pruned_block_region_models = []
        per_block_stats = []

        for k, models_k in enumerate(block_region_models):
            pruned_k, stats_k = self._prune_dominated_regions_one_block(
                models_k=models_k,
                tol=tol,
            )
            stats_k["block_index"] = int(k)
            pruned_block_region_models.append(pruned_k)
            per_block_stats.append(stats_k)

        total_before = sum(s["before"] for s in per_block_stats)
        total_after = sum(s["after"] for s in per_block_stats)
        total_removed = sum(s["removed"] for s in per_block_stats)
        total_pair_tests = sum(s["pair_tests"] for s in per_block_stats)
        total_dominance_hits = sum(s["dominance_hits"] for s in per_block_stats)

        return pruned_block_region_models, {
            "per_block": per_block_stats,
            "total_before": int(total_before),
            "total_after": int(total_after),
            "total_removed": int(total_removed),
            "total_pair_tests": int(total_pair_tests),
            "total_dominance_hits": int(total_dominance_hits),
        }

    def _region_model_min_value(self, rec):
        _, val = self._minimize_sum_of_region_quadratics_over_unit_box(
            alpha_global=np.asarray(rec["alpha_global"], dtype=float),
            beta_global=np.asarray(rec["beta_global"], dtype=float),
            gamma=float(rec["gamma"]),
        )
        return float(val)

    def _sort_block_region_models_for_bb(self, block_region_models):
        sorted_models = []
        for models_k in block_region_models:
            tmp = list(models_k)
            tmp.sort(key=lambda rec: self._region_model_min_value(rec))
            sorted_models.append(tmp)
        return sorted_models

    def _make_greedy_initial_combo_for_bb(self, block_region_models, d_master):
        alpha = np.zeros(d_master, dtype=float)
        beta = np.zeros(d_master, dtype=float)
        gamma = 0.0
        combo = []

        for models_k in block_region_models:
            best_r = 0
            best_single = INFINITY

            for r, rec in enumerate(models_k):
                val_r = self._region_model_min_value(rec)
                if val_r < best_single:
                    best_single = float(val_r)
                    best_r = int(r)

            rec = models_k[best_r]
            alpha += np.asarray(rec["alpha_global"], dtype=float)
            beta += np.asarray(rec["beta_global"], dtype=float)
            gamma += float(rec["gamma"])
            combo.append(best_r)

        t_star, val = self._minimize_sum_of_region_quadratics_over_unit_box(
            alpha_global=alpha,
            beta_global=beta,
            gamma=gamma,
        )

        return {
            "best_val": float(val),
            "best_t": np.asarray(t_star, dtype=float).copy(),
            "best_combo": tuple(int(v) for v in combo),
            "best_alpha": np.asarray(alpha, dtype=float).copy(),
            "best_beta": np.asarray(beta, dtype=float).copy(),
            "best_gamma": float(gamma),
        }

    def _serial_best_combo_search_bb(self, block_region_models, d_master, rank, size):
        """
        Branch-and-bound combo search.
        It still uses closed-form separable quadratic minimization. No solver is called.
        """
        K = len(block_region_models)
        if K == 0:
            raise RuntimeError("TwoLevelQuadratic BB combo search received zero blocks.")

        block_region_models = self._sort_block_region_models_for_bb(block_region_models)

        block_single_mins = []
        for models_k in block_region_models:
            vals_k = [self._region_model_min_value(rec) for rec in models_k]
            block_single_mins.append(float(min(vals_k)))

        suffix_min = [0.0] * (K + 1)
        for k in range(K - 1, -1, -1):
            suffix_min[k] = float(suffix_min[k + 1] + block_single_mins[k])

        init = self._make_greedy_initial_combo_for_bb(block_region_models, d_master)

        best_val = float(init["best_val"])
        best_t = np.asarray(init["best_t"], dtype=float).copy()
        best_combo = init["best_combo"]
        best_alpha = np.asarray(init["best_alpha"], dtype=float).copy()
        best_beta = np.asarray(init["best_beta"], dtype=float).copy()
        best_gamma = float(init["best_gamma"])

        bb_node_count = 0
        bb_pruned_count = 0
        full_combo_count = 0

        prefix_counter = 0

        def recurse(k, alpha, beta, gamma, combo_prefix):
            nonlocal best_val, best_t, best_combo, best_alpha, best_beta, best_gamma
            nonlocal bb_node_count, bb_pruned_count, full_combo_count, prefix_counter

            bb_node_count += 1

            if k == K:
                full_combo_count += 1

                t_star, val = self._minimize_sum_of_region_quadratics_over_unit_box(
                    alpha_global=alpha,
                    beta_global=beta,
                    gamma=gamma,
                )

                if val < best_val:
                    best_val = float(val)
                    best_t = np.asarray(t_star, dtype=float).copy()
                    best_combo = tuple(int(v) for v in combo_prefix)
                    best_alpha = np.asarray(alpha, dtype=float).copy()
                    best_beta = np.asarray(beta, dtype=float).copy()
                    best_gamma = float(gamma)

                return

            for r, rec in enumerate(block_region_models[k]):
                if k == 0:
                    current_prefix_id = prefix_counter
                    prefix_counter += 1
                    if current_prefix_id % int(size) != int(rank):
                        continue

                alpha_new = alpha + np.asarray(rec["alpha_global"], dtype=float)
                beta_new = beta + np.asarray(rec["beta_global"], dtype=float)
                gamma_new = float(gamma + float(rec["gamma"]))

                _, partial_min = self._minimize_sum_of_region_quadratics_over_unit_box(
                    alpha_global=alpha_new,
                    beta_global=beta_new,
                    gamma=gamma_new,
                )

                optimistic_bound = float(partial_min + suffix_min[k + 1])

                if optimistic_bound >= best_val - 1e-12:
                    bb_pruned_count += 1
                    continue

                recurse(
                    k + 1,
                    alpha_new,
                    beta_new,
                    gamma_new,
                    combo_prefix + [r],
                )

        recurse(
            0,
            np.zeros(d_master, dtype=float),
            np.zeros(d_master, dtype=float),
            0.0,
            [],
        )

        return {
            "best_val": float(best_val),
            "best_t": np.asarray(best_t, dtype=float).copy(),
            "best_combo": best_combo,
            "best_alpha": None if best_alpha is None else np.asarray(best_alpha, dtype=float).copy(),
            "best_beta": None if best_beta is None else np.asarray(best_beta, dtype=float).copy(),
            "best_gamma": None if best_gamma is None else float(best_gamma),
            "combo_count": int(full_combo_count),
            "bb_node_count": int(bb_node_count),
            "bb_pruned_count": int(bb_pruned_count),
            "bb_full_combo_count": int(full_combo_count),
        }

    def _print_timing_summary(self, timing):
        if not self.VERBOSE:
            return

        print("=" * 124)
        print("TWO-LEVEL-QUADRATIC LB TIMING SUMMARY (current node)")
        print("=" * 124)
        print(
            f"Mode: {timing['mode']} | MPI enabled: {timing['mpi_enabled']} | "
            f"World size: {timing['world_size']} | d_master: {timing['d_master']}"
        )
        print(
            f"Master samples: {timing['n_master_samples']} | Blocks: {timing['n_blocks']} | "
            f"Requested region tasks: {timing['n_tasks_requested']} | Region models built: {timing['n_region_models_built']}"
        )
        print(
            f"Regions before pruning: {timing['regions_before_pruning']} | "
            f"Regions after pruning: {timing['regions_after_pruning']} | "
            f"Regions removed: {timing['regions_removed_by_pruning']}"
        )
        print(
            f"Combo count before pruning: {timing['combo_count_before_pruning']} | "
            f"Combo count after pruning possible: {timing['combo_count_after_pruning']} | "
            f"Combo count searched/evaluated: {timing['combo_count']}"
        )
        print(
            f"Combo search method: {timing['combo_search_method']} | "
            f"BB threshold: {timing['combo_bb_threshold']}"
        )
        print(
            f"BB nodes visited: {timing['bb_node_count']} | "
            f"BB branches pruned: {timing['bb_pruned_count']} | "
            f"BB full combos evaluated: {timing['bb_full_combo_count']}"
        )
        print(
            f"Pruning pair tests: {timing['pruning_pair_tests']} | "
            f"Dominance hits: {timing['pruning_dominance_hits']}"
        )
        print(
            f"Best combo: {timing['best_combo']} | "
            f"Best original region indices: {timing['best_original_region_indices']} | "
            f"Best LB: {timing['best_lb']:.12g}"
        )

        res = timing.get("residual_correction", None)
        if isinstance(res, dict) and res.get("enabled", False):
            print(
                f"Residual correction: enabled | "
                f"epsilon sum: {res['epsilon_sum']:.6g} | "
                f"epsilon max: {res['epsilon_max']:.6g} | "
                f"h max: {res['h_max']:.6g} | "
                f"Braw max: {res['Braw_max']:.6g} | "
                f"Bhat max: {res['Bhat_max']:.6g} | "
                f"kappa max: {res['kappa_max']:.6g} | "
                f"dN/h_lower max: {res['dN_over_h_lower_max']:.6g} | "
                f"Gamma0: {self.geometry_threshold:.6g} | rho: {res['rho_B']:.6g} | omega: {res['omega']:.6g}"
            )

        print("-" * 124)
        print(f"validate/setup                    : {timing['validate_setup_sec']:.6f} s")
        print(f"normalize block specs             : {timing['normalize_specs_sec']:.6f} s")
        print(f"validate normalized specs         : {timing['validate_specs_sec']:.6f} s")
        print(f"generate region tasks             : {timing['task_generation_sec']:.6f} s")
        print(f"region-model stage total          : {timing['region_model_stage_sec']:.6f} s")
        print(f"flatten/accounting                : {timing['flatten_accounting_sec']:.6f} s")
        print(f"assemble block region models      : {timing['assemble_block_models_sec']:.6f} s")
        print(f"exact block dominance pruning     : {timing['dominance_pruning_sec']:.6f} s")
        print(f"combo-search stage total          : {timing['combo_search_stage_sec']:.6f} s")
        print(f"finalize/pack                     : {timing['finalize_pack_sec']:.6f} s")
        print("-" * 124)
        print(f"TOTAL LB construction time        : {timing['total_lb_construction_sec']:.6f} s")
        print("=" * 124)

    def _underestimate(
        self,
        *,
        all_X,
        bounds,
        all_Y_parts,
        all_block_clouds,
        block_specs,
        sample_per_axis=None,
    ):
        time_start = time.time()

        t0 = time.time()

        if all_X is None:
            raise ValueError("TwoLevelQuadratic: missing all_X.")
        if all_Y_parts is None:
            raise ValueError("TwoLevelQuadratic: missing all_Y_parts.")
        if all_block_clouds is None:
            raise ValueError("TwoLevelQuadratic: missing all_block_clouds.")
        if bounds is None:
            raise ValueError("TwoLevelQuadratic: missing bounds.")

        all_X = np.asarray(all_X, dtype=float)
        all_Y_parts = np.asarray(all_Y_parts, dtype=float)
        bounds = np.asarray(bounds, dtype=float)

        if all_X.ndim != 2:
            raise ValueError("TwoLevelQuadratic: all_X must be 2D.")
        if all_X.shape[0] == 0:
            raise ValueError("TwoLevelQuadratic: all_X must contain at least one sample.")
        if all_Y_parts.ndim != 2:
            raise ValueError("TwoLevelQuadratic: all_Y_parts must be 2D.")
        if all_Y_parts.shape[0] != all_X.shape[0]:
            raise ValueError("TwoLevelQuadratic: row mismatch between all_X and all_Y_parts.")
        if len(all_block_clouds) != all_X.shape[0]:
            raise ValueError("TwoLevelQuadratic: row mismatch between all_X and all_block_clouds.")
        if bounds.ndim != 2 or bounds.shape[0] != 2:
            raise ValueError("TwoLevelQuadratic: bounds must have shape (2, d_master).")
        if all_X.shape[1] != bounds.shape[1]:
            raise ValueError(
                f"TwoLevelQuadratic: all_X dim {all_X.shape[1]} and bounds dim {bounds.shape[1]} mismatch."
            )

        d_master = int(all_X.shape[1])

        if sample_per_axis is not None:
            self.sample_per_axis = int(max(1, sample_per_axis))

        time_validate_setup = time.time() - t0

        t0 = time.time()

        specs = self._normalize_block_specs(block_specs=block_specs)

        time_normalize_specs = time.time() - t0

        t0 = time.time()

        if all_Y_parts.shape[1] < len(specs):
            raise ValueError("TwoLevelQuadratic: all_Y_parts has fewer columns than number of blocks/specs.")

        for k, spec in enumerate(specs):
            idx = list(spec["master_indices"])
            if len(idx) == 0:
                raise ValueError(
                    f"TwoLevelQuadratic: block '{spec['name']}' must depend on at least one master variable."
                )
            bad = [j for j in idx if (j < 0 or j >= d_master)]
            if bad:
                raise ValueError(
                    f"TwoLevelQuadratic: block '{spec['name']}' has invalid master indices {bad} "
                    f"for d_master={d_master}."
                )

        self.last_parts = []
        self.last_total = None

        time_validate_specs = time.time() - t0

        t0 = time.time()

        tasks = []
        for k, spec in enumerate(specs):
            boxes_k = self._get_private_boxes_for_block(spec)
            for r, private_box in enumerate(boxes_k):
                tasks.append({
                    "block_index": int(k),
                    "region_index": int(r),
                    "private_box": tuple(private_box),
                })

        time_task_generation = time.time() - t0

        MPI = _parallel_mpi_if_active()
        world_size = 1
        world_rank = 0
        if MPI is not None:
            world_size = int(MPI.COMM_WORLD.Get_size())
            world_rank = int(MPI.COMM_WORLD.Get_rank())

        use_mpi = (MPI is not None and world_size > 1)
        self.last_mpi = {
            "enabled": bool(use_mpi),
            "requested_tasks": int(len(tasks)),
            "world_size": int(world_size),
            "mode": "controller_worker" if use_mpi else "serial",
        }

        t0 = time.time()

        # Fill-distance values are reused across private partitions whenever
        # the projected coupling sample set is identical within this call.
        self._fill_distance_cache = {}

        if use_mpi:
            if world_rank != 0:
                raise RuntimeError("TwoLevelQuadratic: non-root ranks must not call _underestimate().")

            gathered_payloads = self._parallel_region_models_controller_workers(
                tasks=tasks,
                bounds=bounds,
                all_X=all_X,
                all_Y_parts=all_Y_parts,
                all_block_clouds=all_block_clouds,
                specs=specs,
            )
        else:
            gathered_payloads = [
                self._serial_region_models(
                    tasks=tasks,
                    bounds=bounds,
                    all_X=all_X,
                    all_Y_parts=all_Y_parts,
                    all_block_clouds=all_block_clouds,
                    specs=specs,
                )
            ]

        time_region_model_stage = time.time() - t0

        # Diagnostics only. Summed rank times are NOT MPI elapsed time.
        if self.VERBOSE:
            _profiles = [payload.get("_diagnostic_profile")
                         for payload in gathered_payloads if payload is not None]
            _profiles = [p for p in _profiles if p is not None]
            if _profiles:
                _keys = ("training_sec", "fill_distance_sec", "pyomo_build_sec",
                         "lp_solve_sec", "fit_other_sec", "residual_geometry_sec",
                         "other_sec", "task_total_sec")
                print("\n--- F2 REGIONAL MODEL DIAGNOSTICS (all MPI ranks) ---")
                print(f"Controller region stage elapsed: {time_region_model_stage:.6f} s")
                print(f"Active ranks: {sum(p['task_count'] > 0 for p in _profiles)}"
                      f" / {len(_profiles)}; tasks: {sum(p['task_count'] for p in _profiles)}")
                for _key in _keys:
                    _sum = sum(p["totals"][_key] for p in _profiles)
                    _worst = max(_profiles, key=lambda p: p["totals"][_key])
                    print(f"{_key:24s} SUM={_sum:10.6f} s  "
                          f"MAX-RANK={_worst['totals'][_key]:10.6f} s "
                          f"(rank {_worst['rank']})")
                _slow = max(_profiles, key=lambda p: p["max_task_sec"])
                _busy = max(_profiles, key=lambda p: p["totals"]["task_total_sec"])
                print(f"Slowest individual region: {_slow['max_task_sec']:.6f} s, "
                      f"rank={_slow['rank']}, (block,region)={_slow['max_task_id']}")
                print(f"Most loaded rank: {_busy['rank']}, "
                      f"task count={_busy['task_count']}, "
                      f"task time={_busy['totals']['task_total_sec']:.6f} s")
                print("\n--- F2 RESIDUAL GEOMETRY DEEP DIAGNOSTICS ---")
                _detail = [p for p in _profiles if p.get("geometry_detail")]
                _sum_keys = (
                    "geometry_total_sec", "distance_matrix_sec", "neighbor_sort_sec",
                    "alpha_total_sec", "matrix_rank_sec", "convex_hull_sec",
                    "geometry_calls", "training_points_sum", "neighbor_points_processed",
                    "neighbor_expansions", "alpha_calls", "matrix_rank_calls",
                    "rank_deficient_calls", "convex_hull_calls", "convex_hull_failures",
                )
                for _key in _sum_keys:
                    _total = sum(p["geometry_detail"][_key] for p in _detail)
                    _worst = max(_detail, key=lambda p: p["geometry_detail"][_key])
                    print(f"{_key:27s} SUM={_total:12.6f}  "
                          f"MAX-RANK={_worst['geometry_detail'][_key]:12.6f} "
                          f"(rank {_worst['rank']})")
                for _key in ("max_training_points", "max_joint_dimension", "max_direction_rows",
                             "max_hull_points", "max_hull_facets", "max_selected_neighbors"):
                    _worst = max(_detail, key=lambda p: p["geometry_detail"][_key])
                    print(f"{_key:27s} GLOBAL MAX={_worst['geometry_detail'][_key]} "
                          f"(rank {_worst['rank']})")
                _g = sum(p["geometry_detail"]["geometry_total_sec"] for p in _detail)
                _a = sum(p["geometry_detail"]["alpha_total_sec"] for p in _detail)
                _r = sum(p["geometry_detail"]["matrix_rank_sec"] for p in _detail)
                _h = sum(p["geometry_detail"]["convex_hull_sec"] for p in _detail)
                print(f"Geometry outside alpha helper: {_g - _a:.6f} accumulated rank-seconds")
                print(f"Alpha helper outside rank/hull: {_a - _r - _h:.6f} accumulated rank-seconds")
                print("--- END RESIDUAL GEOMETRY DEEP DIAGNOSTICS ---")
                print("--- END REGIONAL MODEL DIAGNOSTICS ---\n")

        t0 = time.time()

        all_region_models, fevals_this_call, block_fevals_this_call = self._flatten_gathered(gathered_payloads)

        geometry_failures = []
        if self.use_residual_correction:
            for rec in all_region_models:
                if not bool(rec.get("geometry_ok", False)):
                    h_lower = float(rec.get("h_lower_kr", 0.0))
                    dN = float(rec.get("dN_kr", float("inf")))
                    ratio = float("inf") if h_lower <= 0.0 else dN / h_lower
                    geometry_failures.append({
                        "block_name": str(rec.get("block_name", rec.get("block_index", "?"))),
                        "region_index": int(rec.get("region_index", -1)),
                        "kappa": float(rec.get("kappa_kr", float("inf"))),
                        "Gamma0": float(self.geometry_threshold),
                        "dN": dN,
                        "h_lower": h_lower,
                        "h_upper": float(rec.get("h_kr", 0.0)),
                        "ratio": float(ratio),
                    })

        if geometry_failures:
            raise GeometryEnrichmentRequired(geometry_failures)

        eps_vals = [float(rec.get("epsilon_kr", 0.0)) for rec in all_region_models]
        h_vals = [float(rec.get("h_kr", 0.0)) for rec in all_region_models]
        Braw_vals = [float(rec.get("Braw_kr", 0.0)) for rec in all_region_models]
        Bhat_vals = [float(rec.get("Bhat_kr", 0.0)) for rec in all_region_models]
        kappa_vals = [float(rec.get("kappa_kr", 1.0)) for rec in all_region_models]
        ratio_vals = []
        for rec in all_region_models:
            hl = float(rec.get("h_lower_kr", 0.0))
            dn = float(rec.get("dN_kr", 0.0))
            ratio_vals.append(float("inf") if hl <= 0.0 and dn > 0.0 else (0.0 if hl <= 0.0 else dn / hl))

        residual_correction_stats = {
            "enabled": bool(self.use_residual_correction),
            "epsilon_sum": float(np.sum(eps_vals)) if eps_vals else 0.0,
            "epsilon_max": float(np.max(eps_vals)) if eps_vals else 0.0,
            "h_max": float(np.max(h_vals)) if h_vals else 0.0,
            "Braw_max": float(np.max(Braw_vals)) if Braw_vals else 0.0,
            "Bhat_max": float(np.max(Bhat_vals)) if Bhat_vals else 0.0,
            "kappa_max": float(np.max(kappa_vals)) if kappa_vals else 1.0,
            "dN_over_h_lower_max": float(np.max(ratio_vals)) if ratio_vals else 0.0,
            "rho_B": float(self.rho_B),
            "omega": float(self.omega),
        }

        self.last_parts = list(all_region_models)

        self.total_underestimator_fevals += int(fevals_this_call)

        if self.block_underestimator_fevals is None:
            self.block_underestimator_fevals = [0] * len(specs)
        elif len(self.block_underestimator_fevals) < len(specs):
            self.block_underestimator_fevals = self.block_underestimator_fevals + [0] * (
                len(specs) - len(self.block_underestimator_fevals)
            )

        for k, v in enumerate(block_fevals_this_call):
            self.block_underestimator_fevals[k] += int(v)

        time_flatten_accounting = time.time() - t0

        t0 = time.time()

        block_region_models = self._assemble_block_region_models(all_region_models, len(specs))

        for k, models_k in enumerate(block_region_models):
            if len(models_k) == 0:
                raise RuntimeError(f"TwoLevelQuadratic: block {k} received no region models.")

        time_assemble_block_models = time.time() - t0

        combo_count_before_pruning = self._combo_count_from_block_region_models(block_region_models)

        t0 = time.time()

        block_region_models, pruning_stats = self._prune_dominated_regions_by_block(
            block_region_models=block_region_models,
            tol=1e-9,
        )
        self.last_pruning_stats = pruning_stats

        for k, models_k in enumerate(block_region_models):
            if len(models_k) == 0:
                raise RuntimeError(f"TwoLevelQuadratic: block {k} has no region models after pruning.")

        combo_count_after_pruning = self._combo_count_from_block_region_models(block_region_models)

        time_dominance_pruning = time.time() - t0

        t0 = time.time()

        if int(combo_count_after_pruning) > int(self.COMBO_BB_THRESHOLD):
            combo_search_method = "branch_and_bound"

            if use_mpi:
                gathered_best = self._parallel_best_combo_bb_controller_workers(
                    block_region_models=block_region_models,
                    d_master=d_master,
                )
                best_pack = self._select_global_best_combo(gathered_best, d_master=d_master)
            else:
                local_best = self._serial_best_combo_search_bb(
                    block_region_models=block_region_models,
                    d_master=d_master,
                    rank=0,
                    size=1,
                )
                best_pack = self._select_global_best_combo([local_best], d_master=d_master)

        else:
            combo_search_method = "flat_enumeration"

            if use_mpi:
                gathered_best = self._parallel_best_combo_controller_workers(
                    block_region_models=block_region_models,
                    d_master=d_master,
                )
                best_pack = self._select_global_best_combo(gathered_best, d_master=d_master)
            else:
                local_best = self._serial_best_combo_search_implicit(
                    block_region_models=block_region_models,
                    d_master=d_master,
                    rank=0,
                    size=1,
                )
                best_pack = self._select_global_best_combo([local_best], d_master=d_master)

        self.total_combo_evals += int(best_pack["total_combo_count"])

        time_combo_search_stage = time.time() - t0

        t0 = time.time()

        best_val = float(best_pack["best_val"])
        best_t = np.asarray(best_pack["best_t"], dtype=float).copy()
        best_combo = best_pack["best_combo"]
        best_alpha = best_pack["best_alpha"]
        best_beta = best_pack["best_beta"]
        best_gamma = best_pack["best_gamma"]

        best_original_region_indices = None
        if best_combo is not None:
            best_original_region_indices = tuple(
                int(block_region_models[k][r]["region_index"])
                for k, r in enumerate(best_combo)
            )

        self.last_total = {
            "best_combo": best_combo,
            "best_original_region_indices": best_original_region_indices,
            "a_total": None,
            "b_total": None,
            "alpha_total": None if best_alpha is None else np.asarray(best_alpha, dtype=float).copy(),
            "beta_total": None if best_beta is None else np.asarray(best_beta, dtype=float).copy(),
            "gamma_total": None if best_gamma is None else float(best_gamma),
            "best_local_master_point": np.asarray(best_t, dtype=float).copy(),
            "total_combo_count": int(best_pack["total_combo_count"]),
            "combo_count_before_pruning": int(combo_count_before_pruning),
            "combo_count_after_pruning": int(combo_count_after_pruning),
            "combo_search_method": combo_search_method,
            "combo_bb_threshold": int(self.COMBO_BB_THRESHOLD),
            "bb_node_count": int(best_pack.get("total_bb_node_count", 0)),
            "bb_pruned_count": int(best_pack.get("total_bb_pruned_count", 0)),
            "bb_full_combo_count": int(best_pack.get("total_bb_full_combo_count", best_pack["total_combo_count"])),
            "pruning_stats": pruning_stats,
            "residual_correction": residual_correction_stats,
        }

        flb = float(best_val)
        Xnew = np.atleast_2d(np.asarray(best_t, dtype=float))
        lipschitz = 0.0

        time_finalize_pack = time.time() - t0

        total_lb_time = time.time() - time_start
        self.time_underestimate += total_lb_time

        self.last_timing_breakdown = {
            "mode": "controller_worker" if use_mpi else "serial",
            "mpi_enabled": bool(use_mpi),
            "world_size": int(world_size),
            "d_master": int(d_master),
            "n_master_samples": int(all_X.shape[0]),
            "n_blocks": int(len(specs)),
            "n_tasks_requested": int(len(tasks)),
            "n_region_models_built": int(len(all_region_models)),
            "combo_count_before_pruning": int(combo_count_before_pruning),
            "combo_count_after_pruning": int(combo_count_after_pruning),
            "combo_count": int(best_pack["total_combo_count"]),
            "combo_search_method": combo_search_method,
            "combo_bb_threshold": int(self.COMBO_BB_THRESHOLD),
            "bb_node_count": int(best_pack.get("total_bb_node_count", 0)),
            "bb_pruned_count": int(best_pack.get("total_bb_pruned_count", 0)),
            "bb_full_combo_count": int(best_pack.get("total_bb_full_combo_count", best_pack["total_combo_count"])),
            "regions_before_pruning": int(pruning_stats["total_before"]),
            "regions_after_pruning": int(pruning_stats["total_after"]),
            "regions_removed_by_pruning": int(pruning_stats["total_removed"]),
            "pruning_pair_tests": int(pruning_stats["total_pair_tests"]),
            "pruning_dominance_hits": int(pruning_stats["total_dominance_hits"]),
            "best_combo": best_combo,
            "best_original_region_indices": best_original_region_indices,
            "best_lb": float(flb),
            "residual_correction": residual_correction_stats,
            "validate_setup_sec": float(time_validate_setup),
            "normalize_specs_sec": float(time_normalize_specs),
            "validate_specs_sec": float(time_validate_specs),
            "task_generation_sec": float(time_task_generation),
            "region_model_stage_sec": float(time_region_model_stage),
            "flatten_accounting_sec": float(time_flatten_accounting),
            "assemble_block_models_sec": float(time_assemble_block_models),
            "dominance_pruning_sec": float(time_dominance_pruning),
            "combo_search_stage_sec": float(time_combo_search_stage),
            "finalize_pack_sec": float(time_finalize_pack),
            "total_lb_construction_sec": float(total_lb_time),
        }

        self._print_timing_summary(self.last_timing_breakdown)

        return flb, lipschitz, Xnew


