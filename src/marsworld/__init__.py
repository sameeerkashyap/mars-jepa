"""MarsWorld: Physics-Informed JEPA world model for Martian climate dynamics.

This package provides a modular framework for training Joint Embedding
Predictive Architectures (JEPA) with physics-informed objectives on Martian
atmospheric fields. The initial scope targets a *single* physical parameter
(surface temperature) with a synthetic PDE solver, and is designed so that
additional parameters (e.g. dust optical depth) can be added later without
restructuring the codebase.

Sub-packages
------------
- :mod:`marsworld.data`     synthetic PDE solver + torch datasets
- :mod:`marsworld.models`   encoder / predictor / decoder / PI-JEPA model
- :mod:`marsworld.physics`  PDE residual & conservation diagnostics/losses
- :mod:`marsworld.training` trainer loop and composite loss
- :mod:`marsworld.utils`    logging and reproducibility helpers
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
