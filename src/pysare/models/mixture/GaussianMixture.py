import math
import torch
from torch import Tensor
from pysare.models.mixture._MixtureSurvivalModel import MixtureSurvivalModel
from pysare.models.utils.MLP import MLP
from ._mixers._Mixer import SoftmaxMixerWithReflection
from typing import Callable

sqrt2 = math.sqrt(2)
sqrt2pi = math.sqrt(2*math.pi)


class NormalForMixture(torch.nn.Module):

    num_parameters = 2

    def __init__(self, mu_activation=torch.exp, sigma_activation=torch.exp) -> None:
        super().__init__()
        self.mu_activation = mu_activation
        self.sigma_activation = sigma_activation

    def _get_parameters(self, parameters, T):
        Nd = parameters.shape[1]//2
        Nt = len(T)

        T = T.view(-1, 1).expand(Nt, Nd)
        mu = self.mu_activation(parameters[:, ::2])
        sigma = self.sigma_activation(parameters[:, 1::2])

        return mu, sigma, T

    def survival_probability(self, parameters, T):
        mu, sigma, T = self._get_parameters(parameters, T)
        return 0.5 * (1 - torch.erf((T - mu) / (sigma*sqrt2)))

    def lifetime_density(self, parameters, T):
        mu, sigma, T = self._get_parameters(parameters, T)
        return torch.exp(-0.5*((T-mu)/sigma)**2)/(sigma*sqrt2pi)
    
    def forward(self):
        return None


class GaussianMixture(MixtureSurvivalModel):
    r"""Gaussian mixture model with reflection.
    
    Implements a mixture of reflected Gaussian components. Each component is 
    parameterized by a location and scale parameter, corresponding to the mean 
    and standard deviation of the underling gaussian distribution. 

    .. figure:: ../figures/reflected_kernel.pdf
        :class: with-border
        :align: center
        :width: 500px

        Illustration of how a component is reflected around time zero.

    Parameters
    ----------
    location_activation : callable, default=torch.exp
        Activation function for the location parameter. Takes a tensor and
        returns a tensor of the same shape and with non-negative elements.
    scale_activation : callable, default=torch.exp
        Activation function for the scale parameter. Takes a tensor and
        returns a tensor of the same shape and with non-negative elements.
    """
    
    def __init__(self, 
                 location_activation: Callable[[Tensor], Tensor] =torch.exp, 
                 scale_activation: Callable[[Tensor], Tensor] =torch.exp) -> None:
        
        super().__init__(distributions=NormalForMixture(mu_activation=location_activation, 
                                                        sigma_activation=scale_activation),
                         mixer=SoftmaxMixerWithReflection())
    
    def forward_output_size(self, num_components: int):
        return num_components*3

    @classmethod
    def MLP_implementation(cls, num_components, input_size, hidden_sizes,
                           activation=torch.nn.ReLU, dropout=0.0, batch_norm=False,
                           shape_activation=torch.exp, scale_activation=torch.exp):
        return MLPNormalMixture(num_components, input_size, hidden_sizes,
                                activation, dropout, batch_norm,
                                shape_activation, scale_activation)


class MLPNormalMixture(GaussianMixture):
    def __init__(self, num_components, input_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False,
                 location_activation=torch.exp, scale_activation=torch.exp) -> None:
        super().__init__(location_activation=location_activation,
                         scale_activation=scale_activation)

        self.layers = MLP(input_size, num_components*3,
                          hidden_sizes, activation, dropout, batch_norm)
        self.num_components = num_components

    def forward(self, X):
        return self.layers(X)
