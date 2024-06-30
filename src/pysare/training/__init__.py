"""A sub-package for training PySaRe models and other PyTorch models.

The following classes are available:

- :class:`Trainer` -- Used to train models
- :class:`LRFinder`     -- Used to find an appropriate learning rate
- :class:`GridSearch` --  Used for hyperparameter searches
"""

from ._LRFinder import LRFinder
from ._Trainer import Trainer
from ._GridSearch import GridSearch
