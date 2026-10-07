#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBDDSBB was developed using the publicly available PyDDSBB implementation
as a starting codebase. The original PyDDSBB implementation was substantially
modified and extended to implement the proposed decomposition-based framework.

Original PyDDSBB implementation:
https://github.com/DDPSE/PyDDSBB
"""
import os
import time

import numpy as np

from DBDDSBB._core import IID, LHS, Node, Splitter, BoundConstrainedSimulation
import DBDDSBB._DB_underestimator as _DB_underestimator


# Determine rank for root-only printing without importing mpi4py at package import.
# The underestimator loads MPI lazily only when a multi-rank run is active.
def _launcher_rank():
    for name in ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"):
        try:
            return int(os.environ.get(name, ""))
        except ValueError:
            pass
    return 0

_DDSBB_RANK = _launcher_rank()


INFINITY = np.inf


class DecomposedProblemEvaluator:
    """Reusable private-grid evaluation and bookkeeping for decomposed problems."""

    def __init__(
        self,
        block_specs,
        master_dim,
        full_dim,
        m0=20,
        gamma=1.2,
        level_interval=15,
        sample_per_axis=2,
        parallel_function_evaluations=False,
    ):
        self.block_specs = block_specs
        self.master_dim = int(master_dim)
        self.full_dim = int(full_dim)
        self.m0 = int(m0)
        self.gamma = float(gamma)
        self.level_interval = int(level_interval)
        self.sample_per_axis = int(sample_per_axis)
        self.parallel_function_evaluations = bool(parallel_function_evaluations)
        if self.m0 <= 0 or self.level_interval <= 0 or self.sample_per_axis <= 0:
            raise ValueError("m0, level_interval, and sample_per_axis must be positive.")

        self.current_m = self.m0
        self.objective_cache = {}
        self.block_sampling_fevals = [0] * len(self.block_specs)
        self._rebuild_private_grids(self.current_m)

    @staticmethod
    def _uniform_partitions(lb, ub, m):
        pts = np.linspace(float(lb), float(ub), int(m) + 1)
        return [(float(pts[i]), float(pts[i + 1])) for i in range(int(m))]

    @staticmethod
    def _unique_preserve_order(vals, decimals=14):
        out, seen = [], set()
        for v in vals:
            key = round(float(v), decimals)
            if key not in seen:
                seen.add(key)
                out.append(float(v))
        return out

    def _axis_points(self, partitions):
        pts = []
        for lb, ub in partitions:
            pts.extend(np.linspace(float(lb), float(ub), self.sample_per_axis).tolist())
        return np.asarray(self._unique_preserve_order(pts), dtype=float)

    def _build_private_grid(self, partitions_per_axis):
        if len(partitions_per_axis) == 0:
            return np.zeros((1, 0), dtype=float)
        import itertools
        axes = [self._axis_points(parts) for parts in partitions_per_axis]
        return np.asarray(list(itertools.product(*axes)), dtype=float)

    @staticmethod
    def _cache_key(z, decimals=14):
        z = np.asarray(z, dtype=float).ravel()
        return tuple(round(float(v), decimals) for v in z)

    def private_m_for_level(self, level):
        level = int(level)
        if level < 0:
            raise ValueError("Solver tree level must be nonnegative.")
        exponent = level // self.level_interval
        return int(np.ceil(self.m0 * self.gamma ** exponent))

    def _rebuild_private_grids(self, m):
        for spec in self.block_specs:
            bounds = spec.get("private_bounds", [])
            partitions = [self._uniform_partitions(lb, ub, m) for lb, ub in bounds]
            spec["private_partitions"] = partitions
            spec["private_grid"] = self._build_private_grid(partitions)

    def set_private_resolution_for_level(self, level):
        new_m = self.private_m_for_level(level)
        if new_m != self.current_m:
            self._rebuild_private_grids(new_m)
            self.current_m = int(new_m)
            self.objective_cache.clear()
        return self.current_m

    def _build_full_x(self, master_vals, private_best_by_block):
        master_vals = np.asarray(master_vals, dtype=float).ravel()
        x = np.zeros(self.full_dim, dtype=float)
        x[:self.master_dim] = master_vals
        for b, spec in enumerate(self.block_specs):
            idx = spec["private_original_indices"]
            x[idx] = np.asarray(private_best_by_block[b], dtype=float)
        return x

    def objective_parts_with_metadata(self, z):
        z = np.asarray(z, dtype=float).ravel()
        if len(z) != self.master_dim:
            raise ValueError(f"Expected {self.master_dim} master variables, got {len(z)}.")

        key = (self.current_m, self._cache_key(z))
        cached = self.objective_cache.get(key)
        if cached is not None:
            clouds = [{"private_points": c["private_points"].copy(),
                       "block_values": c["block_values"].copy()}
                      for c in cached["block_clouds"]]
            return cached["parts"], cached["x_full"].copy(), clouds

        parts, private_best_by_block, block_clouds = [], {}, []
        for b, spec in enumerate(self.block_specs):
            master_vals = z[spec["master_indices"]]
            sampled = _DB_underestimator.evaluate_private_grid(
                master_values=master_vals,
                block_function=spec["block_function"],
                assemble_args=spec["assemble_args"],
                private_grid=spec["private_grid"],
                parallel_function_evaluations=self.parallel_function_evaluations,
            )
            self.block_sampling_fevals[b] += int(sampled["n_eval"])
            parts.append(float(sampled["best_val"]))
            private_best_by_block[b] = np.asarray(sampled["best_private"], dtype=float).copy()
            block_clouds.append({
                "private_points": np.asarray(sampled["private_points"], dtype=float).copy(),
                "block_values": np.asarray(sampled["block_values"], dtype=float).copy(),
            })

        parts = tuple(parts)
        x_full = self._build_full_x(z, private_best_by_block)
        self.objective_cache[key] = {
            "parts": parts,
            "x_full": x_full.copy(),
            "block_clouds": [{"private_points": c["private_points"].copy(),
                              "block_values": c["block_values"].copy()}
                             for c in block_clouds],
        }
        return parts, x_full.copy(), block_clouds

    def objective_parts(self, z):
        return self.objective_parts_with_metadata(z)[0]

    def objective(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim == 1:
            return float(sum(self.objective_parts(z)))
        return np.asarray([sum(self.objective_parts(row)) for row in z], dtype=float)

    def get_block_sampling_fevals(self):
        return [int(v) for v in self.block_sampling_fevals]

    def get_total_sampling_fevals(self):
        return int(sum(self.block_sampling_fevals))

    def attach_to_problem(self, problem):
        problem.objective_parts = self.objective_parts
        problem.objective_parts_with_metadata = self.objective_parts_with_metadata
        problem.objective_blocks = [[0] for _ in self.block_specs]
        problem.objective_part_names = [spec["name"] for spec in self.block_specs]
        problem.block_specs = self.block_specs
        problem.sample_per_axis = self.sample_per_axis
        problem.m_private = self.current_m
        problem.set_private_resolution_for_level = self.set_private_resolution_for_level
        problem.get_total_sampling_fevals = self.get_total_sampling_fevals
        problem.get_block_sampling_fevals = self.get_block_sampling_fevals
        return problem


class Tree:
    def __init__(self):
        self.Tree = {}
        self.current_level = 0
        self.Tree[self.current_level] = {}
        self.flb_current = INFINITY

        self.yopt_global = INFINITY
        self.xopt_global = None

        self.min_xrange = INFINITY
        self.lipschitz_current = INFINITY

    def _add_level(self):
        self.current_level += 1
        self.Tree[self.current_level] = {}
        self.lowerbound_global = self.flb_current
        self.flb_current = INFINITY
        self._xopt_hist.append(self.xopt_global)
        self.lipschitz = self.lipschitz_current
        self.lipschitz_current = INFINITY

    def _add_node(self, node):
        if self.xopt_global is None and node.xopt_local is not None:
            self.xopt_global = node.xopt_local
            self.best_node = node.node
            self.best_level = node.level
            self.lipschitz = node.lipschitz

        if np.isfinite(node.yopt_local):
            if (not np.isfinite(self.yopt_global)) or (node.yopt_local < self.yopt_global):
                self.yopt_global = float(node.yopt_local)
                self.xopt_global = None if node.xopt_local is None else np.asarray(node.xopt_local, dtype=float).copy()
                self.best_node = node.node
                self.best_level = node.level
                self.lipschitz = node.lipschitz

        if not np.isfinite(node.flb):
            node.set_decision(0)
            self.Tree[self.current_level][node.node] = node
            return

        if node.flb > self.yopt_global:
            node.set_decision(0)
        else:
            if node.yopt_local == INFINITY:
                if node.level == 1:
                    if self.Tree[node.level - 1][node.pn].yopt_local == INFINITY:
                        node.set_decision(0)
                if node.level > 1:
                    parent = self.Tree[node.level - 1][node.pn]
                    if parent.yopt_local == INFINITY and self.Tree[parent.level - 1][parent.pn].yopt_local == INFINITY:
                        node.set_decision(0)
            else:
                node.set_decision(1)
                if node.flb < self.flb_current:
                    self.flb_current = node.flb
                if node.min_xrange < self.min_xrange:
                    self.min_xrange = node.min_xrange
                if node.lipschitz < self.lipschitz_current:
                    self.lipschitz_current = node.lipschitz

        self.Tree[self.current_level][node.node] = node


class NodeOperation:
    def __init__(self, split_method, variable_selection, minimum_bd):
        self.underestimator = _DB_underestimator.TwoLevelQuadratic()
        self._underestimate = self.underestimator._underestimate

        self.split = Splitter(split_method, variable_selection, minimum_bd).split
        self.variable_selection = variable_selection

        self.time_underestimate = 0.0

    def _set_adaptive(self, adaptive_number):
        self.adaptive_number = adaptive_number

    @staticmethod
    def _safe_range(arr):
        arr = np.asarray(arr, dtype=float)
        return np.where(np.isclose(arr, 0.0), 1.0, arr)

    def _adaptive_sample(self):
        # Both formulations retain the two normalized diagonal corners.
        x_corner = np.zeros((2, self.dim), dtype=float)
        x_corner[1, :] = 1.0
        self._update_sample(x_corner)

        if getattr(self, "formulation", "F1") == "F1":
            # Working F1 behavior: adaptive_number is the target total sample count.
            number_new = max(0, int(self.adaptive_number) - int(len(self.y)))
            if number_new > 0:
                Xnew = LHS.augmentLHS(self.X, number_new)
                self._update_sample(Xnew)
            return

        # Working F2 behavior: fresh IID points, with a child-node target and
        # a minimum fresh batch. Geometry-enrichment samples are handled later
        # through _update_sample(), so their private-grid evaluations are counted.
        if self.level == 0:
            number_new = int(max(0, self.adaptive_number))
        else:
            target_multiplier = int(max(1, self.underestimator.iid_target_multiplier))
            min_fresh = int(max(1, self.underestimator.iid_min_fresh_per_node))
            target_total = int(target_multiplier * (2 * self.dim + 1))
            existing = int(len(self.y))
            number_needed = max(0, target_total - existing)
            number_new = max(min_fresh, number_needed)

        if number_new > 0:
            Xnew = IID.initial_sample(self.dim, number_new)
            self._update_sample(Xnew)

    def _min_max_rescaler(self, Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        return Xnew * self.xrange + self.bounds[0, :]

    def _split_node(self, parent):
        child_bound1, child_bound2 = self.split(parent)
        child1 = self._create_child(child_bound1, parent)
        child2 = self._create_child(child_bound2, parent)
        return child1, child2


class BoxConstrained(NodeOperation):
    def __init__(self, split_method, variable_selection, minimum_bd):
        super().__init__(split_method, variable_selection, minimum_bd)

    def _add_problem(self, problem):
        self.simulator = BoundConstrainedSimulation(problem)
        self.bounds = np.asarray(self.simulator._bounds, dtype=float).copy()
        self.dim = int(self.simulator._dim)

        self.objective_part_names = getattr(problem, "objective_part_names", None)
        self.block_specs = getattr(problem, "block_specs", None)

        self.sample_per_axis = getattr(problem, "sample_per_axis", None)
        self.objective_parts_with_metadata = getattr(problem, "objective_parts_with_metadata", None)

        self.get_total_sampling_fevals = getattr(problem, "get_total_sampling_fevals", None)
        self.get_block_sampling_fevals = getattr(problem, "get_block_sampling_fevals", None)

        # Optional problem-side hook for level-dependent private-grid resolution.
        self.problem = problem
        self.set_private_resolution_for_level = getattr(
            problem, "set_private_resolution_for_level", None
        )

    def _set_private_resolution_for_level(self, level):
        """Update optional level-dependent private discretization before sampling."""
        if callable(self.set_private_resolution_for_level):
            self.set_private_resolution_for_level(int(level))

        # The evaluator updates block_specs in place when the private resolution changes.
        self.block_specs = getattr(self.problem, "block_specs", self.block_specs)

    def _default_representative_x(self):
        xc = 0.5 * (self.bounds[0, :] + self.bounds[1, :])
        return np.asarray(xc, dtype=float).reshape(1, -1)

    def _refresh_local_upper_bound_from_stored_parts(self):
        self.yopt_local = INFINITY
        self.xopt_local = self._default_representative_x()

        if len(getattr(self, "valid_ind", [])) == 0:
            return

        vidx = np.asarray(self.valid_ind, dtype=int)

        if getattr(self, "has_parts", False) and self.y_parts is not None:
            totals = np.sum(np.asarray(self.y_parts[vidx, :], dtype=float), axis=1)
        else:
            totals = np.asarray(self.y[vidx], dtype=float)

        finite_mask = np.isfinite(totals)
        if not np.any(finite_mask):
            return

        totals_f = totals[finite_mask]
        vidx_f = vidx[finite_mask]
        j = int(np.argmin(totals_f))
        best_row = int(vidx_f[j])

        self.yopt_local = float(totals_f[j])
        if hasattr(self, "x_full_samples") and self.x_full_samples is not None:
            self.xopt_local = np.asarray(self.x_full_samples[best_row:best_row + 1, :], dtype=float).copy()
        else:
            self.xopt_local = np.asarray(self.x[best_row:best_row + 1, :], dtype=float).copy()

    def _min_max_single_scaler(self):
        y_scalar = float(np.asarray(self.y, dtype=float).ravel()[0])
        self.ymin_local = y_scalar
        self.ymax_local = y_scalar

        self.xrange = np.asarray(self.bounds[1, :] - self.bounds[0, :], dtype=float)
        xrange_safe = self._safe_range(self.xrange)
        self.X = (np.asarray(self.x, dtype=float) - self.bounds[0, :]) / xrange_safe

        self._refresh_local_upper_bound_from_stored_parts()

        self.yrange = self.ymax_local - self.ymin_local
        if self.yrange == 0.0:
            self.Y = np.ones_like(np.asarray(self.y, dtype=float))
        else:
            self.Y = (np.asarray(self.y, dtype=float) - self.ymin_local) / self.yrange

    def _min_max_scaler(self):
        self._refresh_local_upper_bound_from_stored_parts()

        y_arr = np.asarray(self.y, dtype=float)
        if len(self.valid_ind) > 0:
            self.ymin_local = float(np.min(y_arr[self.valid_ind]))
            self.ymax_local = float(np.max(y_arr[self.valid_ind]))
        else:
            self.ymin_local = float(np.min(y_arr))
            self.ymax_local = float(np.max(y_arr))

        self.yrange = self.ymax_local - self.ymin_local
        self.xrange = np.asarray(self.bounds[1, :] - self.bounds[0, :], dtype=float)
        xrange_safe = self._safe_range(self.xrange)

        if self.yrange == 0.0:
            self.Y = np.ones_like(y_arr)
        else:
            self.Y = (y_arr - self.ymin_local) / self.yrange

        self.X = (np.asarray(self.x, dtype=float) - self.bounds[0, :]) / xrange_safe

    def _create_child(self, child_bounds, parent):
        self.level = parent.level + 1
        self._set_private_resolution_for_level(self.level)

        child_bounds = np.asarray(child_bounds, dtype=float).copy()

        ind1 = np.where(np.all(parent.x <= child_bounds[1, :], axis=1))[0]
        ind2 = np.where(np.all(parent.x >= child_bounds[0, :], axis=1))[0]
        ind = np.intersect1d(ind1, ind2)

        self.x = np.asarray(parent.x[ind, :], dtype=float).copy()
        self.y = np.asarray(parent.y[ind], dtype=float).copy()

        if hasattr(parent, "y_parts") and parent.y_parts is not None:
            self.y_parts = np.asarray(parent.y_parts[ind, :], dtype=float).copy()
        else:
            self.y_parts = None

        if hasattr(parent, "x_full_samples") and parent.x_full_samples is not None:
            self.x_full_samples = np.asarray(parent.x_full_samples[ind, :], dtype=float).copy()
        else:
            self.x_full_samples = None

        if hasattr(parent, "block_sample_clouds") and parent.block_sample_clouds is not None:
            self.block_sample_clouds = [parent.block_sample_clouds[i] for i in ind]
        else:
            self.block_sample_clouds = None

        self.valid_ind = [i for i in range(len(ind)) if self.y[i] != INFINITY]
        self.bounds = child_bounds

        self._min_max_scaler()
        self._adaptive_sample()
        flb, lipschitz = self._training_DDCU()

        self.node += 1
        child = Node(parent.level + 1, self.node, self.bounds, parent.node)
        child.add_data(self.x, self.y)
        child.y_parts = None if self.y_parts is None else np.asarray(self.y_parts, dtype=float).copy()
        child.x_full_samples = None if self.x_full_samples is None else np.asarray(self.x_full_samples, dtype=float).copy()
        child.block_sample_clouds = None if self.block_sample_clouds is None else list(self.block_sample_clouds)

        child.set_opt_flb(flb)
        child.set_opt_local(self.yopt_local, self.xopt_local)
        child.set_lipschitz(lipschitz)



        child.add_valid_ind(self.valid_ind)
        return child

    def _update_sample(self, Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim == 1:
            Xnew = Xnew.reshape(1, -1)

        index = [
            i for i in range(len(Xnew))
            if not np.any(np.all(np.round(self.X, 6) == np.round(Xnew[i, :], 6), axis=1))
        ]
        if index == []:
            return

        Xnew = Xnew[index, :]
        xnew = self._min_max_rescaler(Xnew)

        xnew = np.asarray(xnew, dtype=float)
        if xnew.ndim == 1:
            xnew = xnew.reshape(1, -1)

        ynew = self.simulator._simulate(xnew)
        single = (np.asarray(ynew).size == 1)

        xfull_new = None
        block_clouds_new = None

        if getattr(self, "has_parts", False):
            if self.objective_parts_with_metadata is not None:
                if single:
                    parts_i, full_i, clouds_i = self.objective_parts_with_metadata(xnew[0])
                    yparts_new = np.array([parts_i], dtype=float)
                    xfull_new = np.array([full_i], dtype=float)
                    block_clouds_new = [clouds_i]
                else:
                    parts_rows = []
                    full_rows = []
                    cloud_rows = []
                    for xi in xnew:
                        parts_i, full_i, clouds_i = self.objective_parts_with_metadata(xi)
                        parts_rows.append(parts_i)
                        full_rows.append(full_i)
                        cloud_rows.append(clouds_i)
                    yparts_new = np.array(parts_rows, dtype=float)
                    xfull_new = np.array(full_rows, dtype=float)
                    block_clouds_new = cloud_rows
            else:
                raise RuntimeError("Two-Level-Quadratic requires objective_parts_with_metadata.")
        else:
            yparts_new = None

        self.X = np.concatenate((self.X, Xnew), axis=0)
        self.x = np.concatenate((self.x, xnew), axis=0)

        if xfull_new is not None:
            if getattr(self, "x_full_samples", None) is None:
                self.x_full_samples = np.asarray(xfull_new, dtype=float).copy()
            else:
                self.x_full_samples = np.vstack([self.x_full_samples, np.asarray(xfull_new, dtype=float)])

        if block_clouds_new is not None:
            if getattr(self, "block_sample_clouds", None) is None:
                self.block_sample_clouds = list(block_clouds_new)
            else:
                self.block_sample_clouds.extend(block_clouds_new)

        if single:
            ynew = float(np.asarray(ynew).ravel()[0])
            if ynew == -INFINITY:
                raise TypeError("ERROR: Problem Unbounded")

            if ynew != INFINITY:
                self.valid_ind += [len(self.y)]
            self.y = np.append(self.y, ynew)

            if getattr(self, "has_parts", False):
                if self.y_parts is None:
                    self.y_parts = yparts_new.copy()
                else:
                    self.y_parts = np.vstack([self.y_parts, yparts_new])

            if ynew >= self.ymin_local and ynew <= self.ymax_local:
                if self.yrange != 0.0:
                    Ynew = (ynew - self.ymin_local) / self.yrange
                else:
                    Ynew = 1.0
                self.Y = np.append(self.Y, Ynew)

            elif ynew > self.ymax_local:
                if ynew != INFINITY:
                    self.ymax_local = float(ynew)
                self.yrange = self.ymax_local - self.ymin_local
                if self.yrange == 0.0:
                    self.Y = np.ones_like(self.y, dtype=float)
                else:
                    self.Y = (self.y - self.ymin_local) / self.yrange

            elif ynew < self.ymin_local:
                self.ymin_local = float(ynew)
                self.yrange = self.ymax_local - self.ymin_local
                if self.yrange == 0.0:
                    self.Y = np.ones_like(self.y, dtype=float)
                else:
                    self.Y = (self.y - self.ymin_local) / self.yrange

        else:
            ynew = np.asarray(ynew, dtype=float).ravel()
            ymin = float(np.min(ynew))
            if ymin == -INFINITY:
                raise TypeError("ERROR: Problem Unbounded")

            ymax = float(np.max(ynew))
            current = len(self.y)
            valid_ind = [i for i in range(len(ynew)) if ynew[i] != INFINITY]
            if valid_ind != []:
                self.valid_ind += [i + current for i in valid_ind]
            self.y = np.append(self.y, ynew)

            if getattr(self, "has_parts", False):
                if self.y_parts is None:
                    self.y_parts = yparts_new.copy()
                else:
                    self.y_parts = np.vstack([self.y_parts, yparts_new])

            if ymin >= self.ymin_local and ymax <= self.ymax_local:
                if self.yrange != 0.0:
                    Ynew = (ynew - self.ymin_local) / self.yrange
                else:
                    Ynew = np.ones(len(ynew), dtype=float)
                self.Y = np.append(self.Y, Ynew)

            elif ymin >= self.ymin_local and ymax > self.ymax_local:
                if ymax != INFINITY:
                    self.ymax_local = ymax
                self.yrange = self.ymax_local - self.ymin_local
                if self.yrange == 0.0:
                    self.Y = np.ones_like(self.y, dtype=float)
                else:
                    self.Y = (self.y - self.ymin_local) / self.yrange

            elif ymin < self.ymin_local and ymax <= self.ymax_local:
                self.ymin_local = ymin
                self.yrange = self.ymax_local - self.ymin_local
                if self.yrange == 0.0:
                    self.Y = np.ones_like(self.y, dtype=float)
                else:
                    self.Y = (self.y - self.ymin_local) / self.yrange

            elif ymin < self.ymin_local and ymax > self.ymax_local:
                self.ymin_local = ymin
                if ymax != INFINITY:
                    self.ymax_local = ymax
                self.yrange = self.ymax_local - self.ymin_local
                if self.yrange == 0.0:
                    self.Y = np.ones_like(self.y, dtype=float)
                else:
                    self.Y = (self.y - self.ymin_local) / self.yrange

        self._refresh_local_upper_bound_from_stored_parts()

    def _create_root_node(self):
        self.level = 0
        self._set_private_resolution_for_level(self.level)
        self.x = np.asarray(self.bounds, dtype=float).copy()
        self.overallBounds = np.asarray(self.bounds[1, :] - self.bounds[0, :], dtype=float)
        self.node = 0
        self.y = np.asarray(self.simulator._simulate(self.bounds), dtype=float).copy()

        if getattr(self, "has_parts", False):
            if self.objective_parts_with_metadata is not None:
                parts_rows = []
                full_rows = []
                cloud_rows = []
                for xi in self.bounds:
                    parts_i, full_i, clouds_i = self.objective_parts_with_metadata(xi)
                    parts_rows.append(parts_i)
                    full_rows.append(full_i)
                    cloud_rows.append(clouds_i)
                self.y_parts = np.array(parts_rows, dtype=float)
                self.x_full_samples = np.array(full_rows, dtype=float)
                self.block_sample_clouds = cloud_rows
            else:
                raise RuntimeError("Two-Level-Quadratic requires objective_parts_with_metadata.")
        else:
            self.y_parts = None
            self.x_full_samples = None
            self.block_sample_clouds = None

        self.valid_ind = [i for i in range(len(self.y)) if self.y[i] != INFINITY]
        self._min_max_scaler()
        self._adaptive_sample()
        flb, lipschitz = self._training_DDCU()

        root_node = Node(self.level, self.node, self.bounds)
        root_node.add_data(self.x, self.y)
        root_node.y_parts = None if self.y_parts is None else np.asarray(self.y_parts, dtype=float).copy()
        root_node.x_full_samples = None if self.x_full_samples is None else np.asarray(self.x_full_samples, dtype=float).copy()
        root_node.block_sample_clouds = None if self.block_sample_clouds is None else list(self.block_sample_clouds)

        root_node.set_opt_flb(flb)
        root_node.set_opt_local(self.yopt_local, self.xopt_local)
        root_node.set_lipschitz(lipschitz)


        root_node.add_valid_ind(self.valid_ind)


        return root_node

    def _training_DDCU(self):
        time_start = time.time()
        check = 0
        geometry_extra_added = 0

        while True:
            try:
                if not getattr(self, "has_parts", False):
                    raise RuntimeError("Two-Level-Quadratic requires objective_parts/objective_blocks in the call program.")
                if self.y_parts is None:
                    raise RuntimeError("Two-Level-Quadratic requires y_parts but none were stored.")
                if self.block_specs is None:
                    raise RuntimeError("Two-Level-Quadratic requires block_specs in the call program.")
                if self.block_sample_clouds is None:
                    raise RuntimeError("Two-Level-Quadratic requires full stored block_sample_clouds.")

                all_X = np.asarray(self.X[self.valid_ind, :], dtype=float)
                all_Y_parts = np.asarray(self.y_parts[self.valid_ind, :], dtype=float)
                all_block_clouds = [self.block_sample_clouds[i] for i in self.valid_ind]

                flb, lipschitz, Xnew = self._underestimate(
                    all_X=all_X,
                    all_Y_parts=all_Y_parts,
                    all_block_clouds=all_block_clouds,
                    bounds=self.bounds,
                    block_specs=self.block_specs,
                    sample_per_axis=self.sample_per_axis,
                )


                if not np.isfinite(flb):
                    Xnew = np.empty((0, self.dim), dtype=float)

                break

            except _DB_underestimator.GeometryEnrichmentRequired as exc:
                if getattr(self, "formulation", "F1") != "F2":
                    raise

                max_extra = int(max(0, self.underestimator.max_geometry_enrich_samples))
                remaining = max_extra - geometry_extra_added
                if remaining <= 0:
                    raise RuntimeError(
                        "Two-Level-Quadratic geometry condition was not met after "
                        f"{geometry_extra_added} enrichment samples at level={self.level}, "
                        f"node={self.node}.\n{exc}"
                    ) from exc

                multiplier = int(max(1, self.underestimator.iid_geometry_batch_multiplier))
                batch_size = int(max(1, multiplier * self.dim))
                batch_size = min(batch_size, remaining)

                if _DDSBB_RANK == 0:
                    print(
                        "INFO: TLQ geometry enrichment: "
                        f"level={self.level}, node={self.node}, "
                        f"adding {batch_size} fresh IID coupling sample(s); "
                        f"extra so far={geometry_extra_added}/{max_extra}."
                    )
                    print(str(exc))

                Xextra = IID.initial_sample(self.dim, batch_size)
                before = int(len(self.y))
                self._update_sample(Xextra)
                after = int(len(self.y))
                actually_added = max(0, after - before)

                geometry_extra_added += actually_added

                if actually_added == 0:
                    # IID collisions are extraordinarily unlikely, but avoid a
                    # possible infinite loop if duplicate filtering rejects all.
                    geometry_extra_added += batch_size

                continue

            except Exception as exc:
                check += 1
                if check > 20:
                    raise RuntimeError(f"ERROR: Failed to train DDCU. Last error: {exc}")

        if Xnew is not None and np.asarray(Xnew).size > 0:
            self._update_sample(np.asarray(Xnew, dtype=float))

        if np.isfinite(flb) and abs(self.ymin_local - flb) <= 1e-6:
            flb = self.ymin_local

        self.time_underestimate += time.time() - time_start
        return float(flb), float(lipschitz)


class DBDDSBB(Tree):
    def __init__(
        self,
        number_init_samples,
        formulation="F1",
        split_method="equal_bisection",
        variable_selection="longest_side",
        stop_option=None,
        sense="minimize",
        adaptive_sampling=None,
    ):
        super().__init__()
        self.current_level = 0
        self.node = 0
        self.level = 0
        self.stop = 0
        self.sample_number = 0

        self.init_sample = number_init_samples
        self.formulation = str(formulation).upper()
        if self.formulation not in {"F1", "F2"}:
            raise ValueError("formulation must be 'F1' or 'F2'.")
        self.stop_message = "Method Initialized"
        self.split_method = split_method
        self.variable_selection = variable_selection
        if stop_option is None:
            stop_option = {
                "absolute_tolerance": 0.05,
                "relative_tolerance": 0.01,
                "minimum_bound": 0.01,
                "sampling_limit": 10000,
                "time_limit": 36000,
            }

        for key, value in stop_option.items():
            setattr(self, key, value)

        self.total_blockwise_fevals = None
        self.equivalent_blockwise_fevals = None
        self.blockwise_fevals_by_block = None
        self.sampling_fevals_total = None
        self.sampling_fevals_by_block = None
        self.underestimator_fevals_total = None
        self.underestimator_fevals_by_block = None

        if sense == "minimize":
            self.report_LB = "lower bound"
            self.report_UB = "upper bound"
        else:
            self.report_LB = "upper bound"
            self.report_UB = "lower bound"

        if adaptive_sampling is not None:
            self._adaptive = adaptive_sampling

    def print_result(self):
        print(self.stop_message)
        print("Time elapsed: " + str(round(self.time_total, 2)) + "s")
        print("Current level: " + str(self.level))
        print("Current node: " + str(self.builder.node))
        print("Number of samples used: " + str(self.builder.simulator.sample_number))

        if self.total_blockwise_fevals is not None:
            print("Total blockwise function evaluations: " + str(self.total_blockwise_fevals))
        if self.equivalent_blockwise_fevals is not None:
            print("Equivalent blockwise function evaluations: " + str(self.equivalent_blockwise_fevals))

        if self.builder.simulator._sense == "maximize":
            print("Current best " + self.report_LB + " :  " + str(-self.yopt_global))
            print("Current best " + self.report_UB + " :  " + str(-self.lowerbound_global))
        else:
            print("Current best " + self.report_UB + " :  " + str(self.yopt_global))
            print("Current best " + self.report_LB + " :  " + str(self.lowerbound_global))

        gap = self.yopt_global - self.lowerbound_global
        if np.isfinite(gap):
            print("Current absolute gap:  " + str(gap))
        else:
            print("Current absolute gap:  undefined (one of UB/LB is infinite)")

        print("Current best optimizer: " + str(self.xopt_global))

    def get_optimum(self):
        if self.builder.simulator._sense == "maximize":
            return -self.yopt_global
        return self.yopt_global

    def get_optimizer(self):
        return self.xopt_global

    def update_stop_criteria(self, new_stop):
        for key, value in new_stop.items():
            setattr(self, key, value)
        self.stop = 0

    def _has_expandable_nodes(self, level_index):
        if level_index not in self.Tree:
            return False
        for parent in self.Tree[level_index].values():
            if parent.decision == 1 and np.isfinite(parent.flb) and parent.flb <= self.yopt_global:
                return True
        return False

    def _grow(self):
        while self.stop == 0:
            if not self._has_expandable_nodes(self.level):
                self.stop = 6
                self.stop_message = "no active nodes remaining"
                break

            self.level += 1
            self._add_level()
            self.builder._set_adaptive(self._adaptive_rule())
            self._completion_indicator = False

            for parent in self.Tree[self.level - 1].values():
                if parent.decision == 1:
                    if np.isfinite(parent.flb) and parent.flb <= self.yopt_global:
                        child1, child2 = self.builder._split_node(parent)
                        self._add_node(child1)
                        parent.add_child(child1.node)
                        self._add_node(child2)
                        parent.add_child(child2.node)
                    elif parent.flb > self.yopt_global:
                        parent.decision = 0

                self._check_resources()
                if self.stop != 0:
                    self.last_searchednode = parent.node
                    break

            if self.stop == 0:
                self._completion_indicator = True
                self._check_convergence()

    def _continue(self):
        remaining_expandable = False
        for parent in self.Tree[self.level - 1].values():
            if parent.node > self.last_searchednode:
                if parent.decision == 1 and np.isfinite(parent.flb) and parent.flb <= self.yopt_global:
                    remaining_expandable = True
                    child1, child2 = self.builder._split_node(parent)
                    self._add_node(child1)
                    parent.add_child(child1.node)
                    self._add_node(child2)
                    parent.add_child(child2.node)
                elif parent.decision == 1 and parent.flb > self.yopt_global:
                    parent.decision = 0

                self._check_resources()
                if self.stop != 0:
                    self.last_searchednode = parent.node
                    break

        if self.stop == 0:
            self._completion_indicator = True
            if remaining_expandable or len(self.Tree.get(self.level, {})) > 0:
                self._check_convergence()
            else:
                self.stop = 6
                self.stop_message = "no active nodes remaining"

    @staticmethod
    def _adaptive(dim, level):
        return max(int(dim * 11 / level + 3), int(dim * 3) + 3)

    def _adaptive_rule(self):
        return self._adaptive(self.dim, self.level)

    def _update_blockwise_feval_accounting(self):
        sampling_total = 0
        sampling_by_block = None

        f_total = getattr(self.builder, "get_total_sampling_fevals", None)
        if callable(f_total):
            sampling_total = int(f_total())

        f_block = getattr(self.builder, "get_block_sampling_fevals", None)
        if callable(f_block):
            sampling_by_block = [int(v) for v in f_block()]

        under_total = 0
        under_by_block = None

        get_total_fe = getattr(self.builder.underestimator, "get_total_fevals", None)
        if callable(get_total_fe):
            under_total = int(get_total_fe())

        get_block_fe = getattr(self.builder.underestimator, "get_block_fevals", None)
        if callable(get_block_fe):
            tmp = get_block_fe()
            if tmp is not None:
                under_by_block = [int(v) for v in tmp]

        n_blocks = None
        if sampling_by_block is not None:
            n_blocks = len(sampling_by_block)
        if under_by_block is not None:
            if n_blocks is None:
                n_blocks = len(under_by_block)
            else:
                n_blocks = max(n_blocks, len(under_by_block))

        if n_blocks is None:
            block_totals = None
            eq_total = None
        else:
            if sampling_by_block is None:
                sampling_by_block = [0] * n_blocks
            elif len(sampling_by_block) < n_blocks:
                sampling_by_block = sampling_by_block + [0] * (n_blocks - len(sampling_by_block))

            if under_by_block is None:
                under_by_block = [0] * n_blocks
            elif len(under_by_block) < n_blocks:
                under_by_block = under_by_block + [0] * (n_blocks - len(under_by_block))

            block_totals = [int(sampling_by_block[k] + under_by_block[k]) for k in range(n_blocks)]
            eq_total = int(max(block_totals)) if n_blocks > 0 else 0

        total_sum = int(sampling_total + under_total)

        self.sampling_fevals_total = int(sampling_total)
        self.sampling_fevals_by_block = sampling_by_block
        self.underestimator_fevals_total = int(under_total)
        self.underestimator_fevals_by_block = under_by_block
        self.total_blockwise_fevals = total_sum
        self.equivalent_blockwise_fevals = eq_total
        self.blockwise_fevals_by_block = block_totals

    def optimize(self, problem):
        self._lowerbound_hist = []
        self._upperbound_hist = []
        self._xopt_hist = []
        self._sampling_hist = []
        self._cpu_hist = []
        self._lipschitz_hist = []

        self.time_start = time.time()
        self.search_instance = 1
        self.stop_message = "In search process"

        if getattr(problem, "_number_unknown_constraint", 0) != 0 or getattr(problem, "_number_known_constraint", 0) != 0:
            raise ValueError(
                "This solver file is box-constrained only. "
                f"Got unknown_constraints={problem._number_unknown_constraint}, "
                f"known_constraints={problem._number_known_constraint}."
            )

        if getattr(problem, "_dim", None) is None:
            raise ValueError("Problem dimension not defined.")

        self.builder = BoxConstrained(
            self.split_method,
            self.variable_selection,
            self.minimum_bound,
        )

        self.builder._add_problem(problem)
        self.builder.formulation = self.formulation
        self.builder.underestimator.use_residual_correction = (self.formulation == "F2")

        if callable(getattr(problem, "objective_parts", None)) and hasattr(problem, "objective_blocks"):
            self.builder.objective_parts = problem.objective_parts
            self.builder.objective_blocks = problem.objective_blocks
            self.builder.has_parts = True
        else:
            self.builder.has_parts = False

        if hasattr(problem, "objective_part_names"):
            self.builder.objective_part_names = problem.objective_part_names
        if callable(getattr(problem, "objective_parts_with_metadata", None)):
            self.builder.objective_parts_with_metadata = problem.objective_parts_with_metadata

        self.yopt_global = INFINITY

        self.builder._set_adaptive(self.init_sample)
        self.dim = self.builder.dim
        self._add_node(self.builder._create_root_node())
        self._completion_indicator = True

        self._update_blockwise_feval_accounting()

        self._check_convergence()
        if self.stop == 0:
            self._check_resources()
        if self.stop == 0:
            self._grow()

        self.time_total = time.time() - self.time_start

    def resume(self, new_stop_option):
        self.time_elapsed = self.time_total
        self.time_start = time.time()
        self.search_instance += 1
        if _DDSBB_RANK == 0:
            print("Resume search with new resources")
        self.stop_message = "In search process " + str(self.search_instance)
        self.update_stop_criteria(new_stop_option)

        if self._completion_indicator:
            self._grow()
        else:
            self._continue()
            if self._completion_indicator and self.stop == 0:
                self._grow()

        self.time_total = time.time() - self.time_start + self.time_elapsed

    def _check_convergence(self):
        self.lowerbound_global = self.flb_current
        self.lipschitz = self.lipschitz_current
        self._lowerbound_hist.append(self.lowerbound_global)
        self._upperbound_hist.append(self.yopt_global)
        self._sampling_hist.append(self.builder.simulator.sample_number)
        self._lipschitz_hist.append(self.lipschitz)

        if self.search_instance == 1:
            self.time_total = time.time() - self.time_start
        else:
            self.time_total = time.time() - self.time_start + self.time_elapsed

        self._cpu_hist.append(self.time_total)

        if not np.isfinite(self.lowerbound_global):
            return

        if not np.isfinite(self.yopt_global):
            self.stop = 4
            self.stop_message = "Problem Infeasible"
            return

        gap_abs = self.yopt_global - self.lowerbound_global

        if gap_abs <= self.absolute_tolerance:
            self.stop = 1
            self.stop_message = "absolute gap closed"
            return

        if self.lowerbound_global != 0.0:
            gap_rel = gap_abs / abs(self.lowerbound_global)
            if gap_rel <= self.relative_tolerance:
                self.stop = 2
                self.stop_message = "relative gap closed"
                return

        if self.min_xrange <= self.minimum_bound:
            self.stop = 3
            self.stop_message = "search space too small"

    def _check_resources(self):
        if self.search_instance == 1:
            self.time_total = time.time() - self.time_start
        else:
            self.time_total = time.time() - self.time_start + self.time_elapsed

        self.sample_number = self.builder.simulator.sample_number
        self._update_blockwise_feval_accounting()

        if hasattr(self, "max_feval_total_limit") and self.total_blockwise_fevals is not None:
            if self.total_blockwise_fevals >= self.max_feval_total_limit:
                self.stop_message = "reached max total function evaluation limit"
                self.stop = 7
                if self.stop != 0 and self._completion_indicator is False:
                    self.stop_message += " without search all active nodes in level " + str(self.level)
                return

        if hasattr(self, "max_feval_equivalent_limit") and self.equivalent_blockwise_fevals is not None:
            if self.equivalent_blockwise_fevals >= self.max_feval_equivalent_limit:
                self.stop_message = "reached max equivalent function evaluation limit"
                self.stop = 8
                if self.stop != 0 and self._completion_indicator is False:
                    self.stop_message += " without search all active nodes in level " + str(self.level)
                return

        if self.sample_number >= self.sampling_limit:
            self.stop_message = "reached sampling limit"
            self.stop = 4
        else:
            if self.time_total >= self.time_limit:
                self.stop_message = "reached time limit"
                self.stop = 5

        if self.stop != 0 and self._completion_indicator is False:
            self.stop_message = self.stop_message + " without search all active nodes in level " + str(self.level)
