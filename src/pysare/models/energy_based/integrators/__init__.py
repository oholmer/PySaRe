r"""Integrators to be used in energy based models.

The following integrators are available:

.. autosummary:: 
    :toctree: generated
    :recursive:

    RandomMonteCarlo
    EquidistantTrapezoidal
    AdaptiveTrapezoidalSimpsons

"""

from ._Integrator import RandomMonteCarlo
from ._Integrator import EquidistantTrapezoidal
from ._Integrator import AdaptiveTrapezoidalSimpsons