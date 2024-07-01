r"""Implementations and tools for defining survival models. 

For an introduction on how to define and use models, see 
https://github.com/oholmer/PySaRe/blob/main/Introduction.ipynb.

Neural Network-Based Models
---------------------------
The following neural network-based models are available as submodules:

-  :mod:`pysare.models.piecewise` -- Piecewise defined models
-  :mod:`pysare.models.mixture` -- Mixture models
-  :mod:`pysare.models.energy_based` -- Energy based models
-  :mod:`pysare.models.distributions` -- Standard distributions


Model Functionality
-------------------     
PySaRe models extend PyTorch modules to provide the following methods:

- :py:meth:`SurvivalModel.survival_probability`
- :py:meth:`SurvivalModel.lifetime_density`
- :py:meth:`SurvivalModel.hazard_function`
- :py:meth:`SurvivalModel.log_likelihood`


NumPy Interface
---------------
PyTorch module requires the inputs to be a tensor of correct dtype and device, 
and to simply the usage of models a model can be converted to handle any inputs
that can be converted to numpy Arrays. This is done using the :func:`to_numpy`
method:

    >>> np_model = model.to_numpy()

Frozen Models
-------------
Predictions are often done multiple times for the same covariates/features and 
a model can therefore be frozen at a specific covariate creating a frozen model 
that does not require the covariate as input. This is done using the 
:meth:`feeze` method:      

    >>> frozen_model = model.freeze(X[0])


Importing Models
----------------

- :func:`pysare.model.from_survival_probabilities`
    Creates a model from a list of times and corresponding survival probabilities.
- :func:`pysare.model.from_survival_function`
    Creates a spline approximation of a given survival function.

The following example creates an approximation of a Weibull distribution:
    
    >>> model = pysare.models.from_survival_function(lambda t: math.exp(-t**2))  

Frozen models can also be created directly from a list of survival probabilities
and can thus be used as a way to use models from other sources in PySaRe.

    >>> frozen_model = pysare.models.from_survival_probabilities(times, probs)

"""

from . import _numpy
from . import piecewise
from . import distributions
from . import energy_based
from . import mixture
from . import utils

from .ensemble import ensemble

from ._KaplanMeier import KaplanMeierEstimator

from .SurvivalModel import TorchSurvivalModel
# from . import energy_based


