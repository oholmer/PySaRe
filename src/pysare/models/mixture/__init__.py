r"""Mixture models

Models based on a mixture of multiple components. The parameters of each 
component and the weights in the mixture are determined based on the 
output from the forward function, see :ref:`Forward output <forward-output>`
below.

The following models are available:

- :class:`WeibullMixture`
    A mix of Weibull distributions. Two parameters per component,
    shape and scale.
- :class:`GaussianMixture`
    A mix of Gaussian/Normal distributions, reflected around zero. 
    Two parameters per component, location and variance.

.. _forward-output:

Forward output
--------------

Let N be the number of components in the mixture and M the number of 
parameters for each component. A row of the output from forward should 
have the format `[p_1_1, ..., p_1_M, p_2_1, ...,P_N_M, w_1, ..., w_1]`
where p_i_j is the value of parameter j in component i, before activation;
and w_1,..., w_n are the mixing weights, before softmax. The size of the 
forward output should therefore be (batch_size, (M+1)*N).

"""

from . import _mixers
from .GaussianMixture import GaussianMixture
from .WeibullMixture import WeibullMixture