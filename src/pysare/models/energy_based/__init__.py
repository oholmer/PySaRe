r"""Energy based models.

In an energy based model, a normalized neural network is used to direcly specify the 
so density. The normalization of the network is done using an
integrator from :mod:`pysare.model.energy_based.integrators`.

The model itself is implemented in :class:`EnergyBasedDensity`.

"""

from .EnergyBasedDensity import EnergyBasedDensity
from . import integrators