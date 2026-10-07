# DBDDSBB

**Decomposition-Based Data-Driven Spatial Branch-and-Bound (DBDDSBB)** is a
Python implementation of a decomposition-based framework for data-driven
global optimization of black-box problems with partially separable structure.

The method exploits problem structure by decomposing the objective into
lower-dimensional blocks while coordinating the blocks through shared
coupling variables. Two lower-bounding formulations, F1 and F2, are
implemented.

## Acknowledgment

DBDDSBB was developed using the publicly available
[PyDDSBB](https://github.com/DDPSE/PyDDSBB) implementation as a starting
codebase. The original PyDDSBB implementation was substantially modified
and extended to implement the proposed decomposition-based framework.

## License

See the `LICENSE` file for licensing information.
