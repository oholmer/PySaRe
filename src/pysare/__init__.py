r"""PySaRe is a package for Data-Driven Survival Analysis and Reliability Engineering based on PyTorch.

Submodules
----------
- ``models`` -- Implementations of survival models
- ``data`` -- Tools for handling data and some included datasets
- ``training`` -- Tools for training models 
- ``evaluate`` -- Evaluation metrics
"""

from . import data
from . import models
from . import training
from . import reliability
from . import evaluation
from . import age_replacement