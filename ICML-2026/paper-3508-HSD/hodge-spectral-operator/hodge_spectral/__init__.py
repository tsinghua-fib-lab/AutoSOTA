"""
Hodge Spectral Operator (HSO)
==============================
Neural operator learning on manifolds with Hodge decomposition inductive bias.

Supports any geometric input: mesh, point cloud, graph.
Supports any differential form task: 0-form→0-form, 0-form→1-form, etc.

Quick start:
    from hodge_spectral import HodgeOperator

    model = HodgeOperator.from_mesh(points, faces, task="0to1")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
"""

__version__ = "0.1.0"

from hodge_spectral.api import HodgeOperator

__all__ = ["HodgeOperator"]
