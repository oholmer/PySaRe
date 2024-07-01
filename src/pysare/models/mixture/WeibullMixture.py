import torch
from pysare.models.mixture._MixtureSurvivalModel import MixtureSurvivalModel
from pysare.models.utils.MLP import MLP
from ._mixers._Mixer import SoftmaxMixer
from typing import Callable
from torch import Tensor



class WeibullForMixture(torch.nn.Module):

    num_parameters = 2

    def __init__(self, shape_activation=torch.exp, scale_activation=torch.exp) -> None:
        super().__init__()
        self.shape_activation = shape_activation
        self.scale_activation = scale_activation

    def _get_parameters(self, parameters, T):
        Np = 2
        Nd = parameters.shape[1]//2
        Nt = len(T)

        T = T.view(-1, 1).expand(Nt, Nd)
        shape = self.shape_activation(parameters[:, ::2])
        scale = self.scale_activation(parameters[:, 1::2])

        return shape, scale, T

    def survival_probability(self, parameters, T):
        shape, scale, T = self._get_parameters(parameters, T)
        return torch.exp(- (T/scale)**shape)

    def lifetime_density(self, parameters, T):
        shape, scale, T = self._get_parameters(parameters, T)
        return (shape/scale)*(T/scale)**(shape-1)*torch.exp(- (T/scale)**shape)


class WeibullMixture(MixtureSurvivalModel):
    r"""Weibull mixture model.
    
    Implements a mixture of Weibull components. Each component is parameterized 
    by a shape and scale parameter, and have the survival function 
    `S(t) = exp((t/scale)**shape)`
     

    For a model with M components, each row in the forward function should be 
    on the form 
    `[shape_1, scale_1, ..., shape_M, scale_M, weight_1, ... weight_M]`
    where shape_i, scale_i are the shape and scale parameters for component i, 
    before activation; and weight_1,... weight_M are the weights before 
    activation.

    Parameters
    ----------

    shape_activation : callable, default=torch.exp
        Activation function for the shape parameter. Takes a tensor and
        returns a tensor of the same shape and with non-negative elements.
    scale_activation : callable, default=torch.exp
        Activation function for the scale parameter. Takes a tensor and
        returns a tensor of the same shape and with non-negative elements.
    """
    def __init__(self, 
                 shape_activation:Callable[[Tensor], Tensor]=torch.exp, 
                 scale_activation:Callable[[Tensor], Tensor]=torch.exp) -> None:

        super().__init__(distributions=WeibullForMixture(
            shape_activation=torch.exp, scale_activation=torch.exp), mixer=SoftmaxMixer())


    @classmethod
    def MLP_implementation(cls, num_components, input_size, hidden_sizes,
                           activation=torch.nn.ReLU, dropout=0.0, batch_norm=False,
                           shape_activation=torch.exp, scale_activation=torch.exp):
        return MLPWeibullMixture(num_components, input_size, hidden_sizes,
                                 activation, dropout, batch_norm,
                                 shape_activation, scale_activation)

    def forward_output_size(self, num_components: int):
        return num_components*3

class MLPWeibullMixture(WeibullMixture):
    def __init__(self, num_components, input_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False,
                 shape_activation=torch.exp, scale_activation=torch.exp) -> None:
        super().__init__(shape_activation=shape_activation,
                         scale_activation=scale_activation)

        self.layers = MLP(input_size, num_components*3,
                          hidden_sizes, activation, dropout, batch_norm)
        self.num_components = num_components

    def forward(self, X):
        return self.layers(X)

# %%