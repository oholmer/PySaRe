r"""Piecewise defined models.

The piecewise models use a discretization grid from 
:mod:`pysare.models.piecewise.grid` and and a neural network 
to specify a piecewise model.  

The following models are available:

- :class:`ConstantDensity`: 
    Piecewise constant density (num_intervals+1)
- :class:`LinearDensity`: 
    Piecewise linear density (num_intervals+2)
- :class:`ConstantHazard`: 
    Piecewise constant hazard (num_intervals)
- :class:`LinearHazard`: 
    Piecewise linear hazard (num_intervals+1)

where the number in parenthesis is the required size of the output
from the forward method, if the chosen grid does not require extra
outputs. The correct output size can also be determined using
the method forward_output_size().

Below is an example of a :class:`LinearDensity` model with a 
:class:`~grid.EquidistantGrid`:

    >>> from pysare.models.piecewise import LinearDensity
    >>> from pysare.models.piecewise.grid import UniformGrid
    ...
    >>> class Model(LinearDensity):
    ...     def __init__(self):
    ...         super().__init__(grid=UniformGrid(max_time, num_intervals))
    ...         self.net = ...
    ...         self.final_layer = torch.nn.Linear(...,
    ...                                       self.forward_output_size())
    ...
    ...     def forward(self, X):
    ...         return self.final_layer(self.net(X))

"""

from ._ConstantDensity import ConstantDensity
from ._LinearDensity import LinearDensity
from ._ConstantHazard import ConstantHazard
from ._LinearHazard import LinearHazard
from . import grid
