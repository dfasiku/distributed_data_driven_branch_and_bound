#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBDDSBB was developed using the publicly available PyDDSBB implementation
as a starting codebase. The original PyDDSBB implementation was substantially
modified and extended to implement the proposed decomposition-based framework.

Original PyDDSBB implementation:
https://github.com/DDPSE/PyDDSBB
"""
from __future__ import annotations

import time
import numpy as np


# =============================================================================
# Problem definition
# =============================================================================

class Problem:
    """Minimal box-constrained problem container used by DBDDSBB."""

    def __init__(self):
        self._dim = 0
        self._number_known_constraint = 0
        self._number_unknown_constraint = 0
        self._variable = []
        self._objective = None
        self._sense = "minimize"

    def add_variable(self, lb, ub, vartype="continuous"):
        if str(vartype).lower() not in {"continuous", "real", "float"}:
            raise ValueError(
                "DBDDSBB currently supports continuous variables only."
            )

        variable = _Variable(lb, ub)
        self._variable.append(variable)
        self._dim = len(self._variable)

    def add_objective(self, objective, sense="minimize"):
        if not callable(objective):
            raise TypeError("objective must be callable.")

        sense = str(sense).lower()

        if sense not in {"minimize", "maximize"}:
            raise ValueError(
                "sense must be 'minimize' or 'maximize'."
            )

        self._objective = objective
        self._sense = sense

    def add_known_constraint(self, *args, **kwargs):
        raise NotImplementedError(
            "DBDDSBB is box-constrained; known constraints are not supported."
        )

    def add_unknown_constraint(self, *args, **kwargs):
        raise NotImplementedError(
            "DBDDSBB is box-constrained; unknown constraints are not supported."
        )


class _Variable:
    """Minimal continuous-variable representation."""

    def __init__(self, lb, ub):
        self._bounds = np.asarray(
            [float(lb), float(ub)],
            dtype=float,
        )


# =============================================================================
# Bound-constrained simulation
# =============================================================================

class BoundConstrainedSimulation:
    """
    Objective wrapper preserving the original DDSBB evaluation and
    sample-count semantics.
    """

    problem_type = "BoundConstrained"

    def __init__(self, problem):
        self._dim = problem._dim

        self._bounds = np.array(
            [i._bounds for i in problem._variable]
        ).T

        self._objective = problem._objective
        self._sense = problem._sense

        self.time_sampling = 0.0

        if self._sense == "minimize":
            self._simulate = self._obj_minimize
        elif self._sense == "maximize":
            self._simulate = self._obj_maximize
        else:
            raise ValueError(
                "Objective sense must be 'minimize' or 'maximize'."
            )

        self.sample_number = 0

    def _obj_maximize(self, x):
        time_start = time.time()

        x = np.asarray(x)

        # Single point: shape (dim,)
        if x.ndim == 1:
            self.sample_number += 1

            y = -float(
                self._objective(x)
            )

            self.time_sampling += (
                time.time() - time_start
            )

            return y

        # Batch of points: shape (n, dim)
        self.sample_number += x.shape[0]

        y = np.array(
            [
                -self._objective(x[j, :])
                for j in range(x.shape[0])
            ],
            dtype=float,
        )

        self.time_sampling += (
            time.time() - time_start
        )

        if len(y) == 1:
            return float(y)

        return y

    def _obj_minimize(self, x):
        time_start = time.time()

        x = np.asarray(x)

        # Single point: shape (dim,)
        if x.ndim == 1:
            self.sample_number += 1

            y = float(
                self._objective(x)
            )

            self.time_sampling += (
                time.time() - time_start
            )

            return y

        # Batch of points: shape (n, dim)
        self.sample_number += x.shape[0]

        y = np.array(
            [
                self._objective(x[j, :])
                for j in range(x.shape[0])
            ],
            dtype=float,
        )

        self.time_sampling += (
            time.time() - time_start
        )

        if len(y) == 1:
            return float(y)

        return y


# =============================================================================
# Branch-and-bound node
# =============================================================================

class Node:
    """
    Branch-and-bound node.

    Retains the active state required by the decomposition-based DDSBB
    implementation.
    """

    def __init__(self, level, node, bounds, pn=None):
        self.node = node
        self.level = level
        self.bounds = bounds

        self.xrange = (
            self.bounds[1, :]
            - self.bounds[0, :]
        )

        # Historical DDSBB name:
        # this is actually the maximum side width.
        self.min_xrange = max(self.xrange)

        self.decision = 1
        self.child = []
        self.pn = pn

    def add_child(self, node):
        self.child.append(node)

    def add_parent(self, node):
        self.pn = node

    def add_data(self, x, y):
        self.x = x
        self.y = y

    def set_opt_local(self, fub, xopt):
        self.yopt_local = fub
        self.xopt_local = xopt

    def set_opt_flb(self, flb):
        self.flb = flb

    def set_lipschitz(self, lipschitz):
        self.lipschitz = lipschitz

    def set_decision(self, decision):
        self.decision = decision

    def add_valid_ind(self, valid_ind):
        self.valid_ind = valid_ind


# =============================================================================
# Splitter
# =============================================================================

class Splitter:
    """
    Active DBDDSBB splitting rule:

        variable selection : longest side
        split location     : equal bisection
    """

    def __init__(
        self,
        split_method="equal_bisection",
        variable_selection="longest_side",
        minimum_bd=0.05,
    ):
        if split_method != "equal_bisection":
            raise ValueError(
                "DBDDSBB supports only "
                "split_method='equal_bisection'."
            )

        if variable_selection != "longest_side":
            raise ValueError(
                "DBDDSBB supports only "
                "variable_selection='longest_side'."
            )

        self.minimum_bd = minimum_bd

    def split(self, parent):
        """
        Preserve the original PyDDSBB child ordering exactly.
        """

        split_x = self.longest_side(parent)
        split_spot = self.equal_bisection(
            parent,
            split_x,
        )

        child_bound1 = parent.bounds.copy()
        child_bound2 = parent.bounds.copy()

        # IMPORTANT:
        # Original PyDDSBB ordering is retained.
        #
        # child 1 = upper half
        # child 2 = lower half
        child_bound1[0, split_x] = split_spot
        child_bound2[1, split_x] = split_spot

        return child_bound1, child_bound2

    @staticmethod
    def equal_bisection(parent, split_x):
        return (
            0.5
            * (
                parent.bounds[1, split_x]
                - parent.bounds[0, split_x]
            )
            + parent.bounds[0, split_x]
        )

    @staticmethod
    def longest_side(parent):
        return np.argmax(
            parent.bounds[1, :]
            - parent.bounds[0, :]
        )


# =============================================================================
# Formulation 2: IID sampling
# =============================================================================

class IID:
    """
    Formulation 2 sampling.

    Fresh coupling samples are drawn independently and uniformly from
    the normalized coupling domain [0, 1]^dim.
    """

    @staticmethod
    def initial_sample(dim, number_new_points):
        """
        Generate fresh IID uniform points in [0,1]^dim.
        """

        dim = int(dim)
        number_new_points = int(number_new_points)

        if dim <= 0:
            raise ValueError(
                "dim must be positive."
            )

        if number_new_points < 0:
            raise ValueError(
                "number_new_points must be nonnegative."
            )

        if number_new_points == 0:
            return np.empty(
                (0, dim),
                dtype=float,
            )

        return np.random.uniform(
            0.0,
            1.0,
            size=(number_new_points, dim),
        )

    @staticmethod
    def augmentIID(
        original_design,
        number_new_points,
    ):
        """
        Generate additional fresh IID uniform points in [0,1]^dim.

        Existing points determine only the dimension.
        """

        original_design = np.asarray(
            original_design,
            dtype=float,
        )

        if original_design.ndim != 2:
            raise ValueError(
                "original_design must be a 2D array."
            )

        return IID.initial_sample(
            original_design.shape[1],
            number_new_points,
        )


# =============================================================================
# Formulation 1: LHS sampling
# =============================================================================

class LHS:
    """
    Original DDSBB LHS augmentation used by Formulation 1.

    The numerical implementation below is intentionally retained from
    the working PyDDSBB implementation to preserve the random-number
    sequence and sampling behavior.
    """

    @staticmethod
    def initial_sample(dim, number_new_points):
        """
        Generate the initial LHS design.
        """

        current_design = np.array(
            [
                [
                    np.random.uniform(0, 1)
                    for i in range(dim)
                ]
            ]
        )

        new_design = LHS.augmentLHS(
            current_design,
            number_new_points - 1,
        )

        return np.vstack(
            (
                current_design,
                new_design,
            )
        )

    @staticmethod
    def augmentLHS(
        original_design,
        number_new_points,
    ):
        """
        Augment an existing LHS design.

        This preserves the original PyDDSBB implementation.
        """

        number_old_points, dim = (
            original_design.shape
        )

        # -------------------------------------------------------------
        # Generate cells for LHS design
        # -------------------------------------------------------------

        number_cells = (
            number_old_points
            + number_new_points
        ) ** 2

        cell_size = 1.0 / (
            number_cells + 1
        )

        cell_lo = [
            i * cell_size
            for i in range(number_cells + 1)
        ]

        cell_up = [
            (i + 1) * cell_size
            for i in range(number_cells + 1)
        ]

        number_candidate_points = (
            number_new_points * 2
        )

        # -------------------------------------------------------------
        # Find filled cells
        # -------------------------------------------------------------

        def find_filled(design):
            filtered = filter(
                lambda x: x >= 0,
                design - cell_lo,
            )

            return len(
                list(filtered)
            ) - 1

        # -------------------------------------------------------------
        # Randomized candidate design in unfilled cells
        # -------------------------------------------------------------

        def select_candidates(col_vec):
            allfilled = list(
                map(
                    find_filled,
                    col_vec,
                )
            )

            candidate_cells = np.random.choice(
                [
                    k
                    for k
                    in range(number_cells + 1)
                    if k not in allfilled
                ],
                number_candidate_points,
            )

            return [
                float(
                    np.random.uniform(
                        cell_lo[k],
                        cell_up[k],
                        1,
                    )
                )
                for k in candidate_cells
            ]

        candidate_points = [
            select_candidates(
                original_design[:, i]
            )
            for i in range(dim)
        ]

        candidate_points_filtered = (
            np.array(
                candidate_points
            ).T
        )

        new_points = []

        current_design = (
            original_design.copy()
        )

        candidates = (
            candidate_points_filtered.copy()
        )

        distance = np.min(
            np.sum(
                np.square(
                    current_design[:, None]
                    - candidates
                ),
                2,
            ),
            0,
        )

        # -------------------------------------------------------------
        # Adaptively add max-min-distance points
        # -------------------------------------------------------------

        while (
            len(new_points)
            != number_new_points
        ):
            selected = np.argmax(
                distance
            )

            new_points.append(
                candidates[selected, :]
            )

            current_design = np.concatenate(
                (
                    original_design,
                    new_points,
                ),
                axis=0,
            )

            candidates = np.delete(
                candidates,
                selected,
                0,
            )

            distance = np.delete(
                distance,
                selected,
                0,
            )

            distance = np.array(
                [
                    min(
                        np.sum(
                            (
                                np.array(
                                    current_design[-1, :]
                                )
                                - candidates[i, :]
                            )
                            ** 2
                        ),
                        distance[i],
                    )
                    for i
                    in range(
                        len(candidates)
                    )
                ]
            )

        new_design = np.array(
            new_points
        )

        return new_design