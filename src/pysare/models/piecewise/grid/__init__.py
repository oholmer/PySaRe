"""Discretization grids to be used with piecewise models.

The following grids are available:

- :class:`EquidistantGrid`
    Defines a fixed equidistant grid.
- :class:`FixedGrid`
    Defines a fixed grid based on a supplied list of grid points.
- :class:`TrainableGrid`
    Defines a grid where the grid points are trainable. 
- :class:`FixedGrid`
    Defines a grid where the grid points are determined based on the 
    output from the models forward function.
    
A custom grid can be defined by subclassing the base class :class:`DiscretizationGrid`.

.. autosummary:: 
    :toctree: generated
    :recursive:

    EquidistantGrid
    IndividualizedGrid
    FixedGrid
    TrainableGrid    
    DiscretizationGrid

"""

from ._grid import EquidistantGrid
from ._grid import DiscretizationGrid
from ._grid import IndividualizedGrid
from ._grid import FixedGrid
from ._grid import TrainableGrid