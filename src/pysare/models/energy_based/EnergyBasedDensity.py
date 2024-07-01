import numpy as np
import torch
from torch import nn
import torch
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models.utils import MLP
from pysare.models.energy_based.integrators._Integrator import EnergyIntegrator, EquidistantTrapezoidal, RandomMonteCarlo
from torch.nn import Module
from typing import Optional, Union

class EnergyBasedDensity(TorchSurvivalModel):
    r"""Implements an energy based survival model.

    In this model, the density is directly specified using a normalized
    neural network. The unnormalized output from the network is called the 
    energy, hence the name. See [1] for more information.

    The model uses two networks, one in the forward function and one to
    calculate the energy:

    - Forward network
        This network is implemented in the forward function 
        and takes the covariates/features X an should return a tensor of shape 
        `(batch_size, energy_input_size-1)` where batch_size is number of 
        samples in the batch (size of first dimension of X) and
        `energy_input_size` is the input size of the energy network.
        By default the forward network implements the identity function (that is, 
        `forward(X)=X`).
    - Energy network
        The energy network is used to calculate the energy based
        on the output from the  forward network. The input to the energy network
        is the output from the forward network concatenated with a tensor of the 
        times of interest. The output from the energy network should be a tensor of 
        shape (batch_size,1) with numbers in (-inf, inf). 

    This model can either be subclassed or instanced. If instanced, a network 
    must be passed as the parameter energy_net and the forward function will
    be the identity function (that is, `forward(X)=X`).

    The normalization of the energy network is done using an integrator 
    from :mod:`pysare.models.energy_based.integrators`. The model only 
    explicitly model the density up to a specified maximal time, specified
    by the parameter `max_time`. The remaining tail is modeled using a single 
    point to after `max_time`, whose location is determined by the parameter
    `tail_ratio` as `tail_ratio*max_time`. See the figure below for an 
    illustration.  

    .. figure:: ../figures/tail_illustration.pdf
        :class: with-border
        :align: center
        :width: 500px

        Illustration of how the tail is treated in the integration. The blue 
        area is approximated using numerical integration and the tail is 
        modeled as the red area, calculated using a single point. Here :math:`t_m` 
        corresponds to `max_time` and :math:`\gamma` to 
        `tail_ratio`. See [1] for details.

    Parameters:
    -----------

    max_time : float
        Maximal prediction time. 
    tail_ratio : float, default=1.2
        Float that determines with point is used to calculate the integral of 
        the tail of the energy network. The tail is approximated as 
        `tail_energy * max_time * (1- tail_ratio)`
        where `tail_energy` is the energy at time `max_time*tail_ratio`. 
    train_integrator : EnergyIntegrator or int
        Integrator used during training. Set as active integrator after 
        construction and automatically set when the model is set to training mode 
        using the `train` method. Can be changed using the `set_integrators` 
        method. If and integer is provided, `RandomMonteCarlo` with the 
        specified number of samples is used.
    eval_integrator : EnergyIntegrator or int
        Integrator used during evaluation. Automatically set when the model is 
        set to evaluation mode using the `eval` method. Can be changed using 
        the `set_integrators` method. If and integer is provided, 
        `EquidistantTrapezoidal` with the specified number of samples 
        is used.
    energy_net : Module, optional
        If supplied this net will be used as energy. The `energy_net.forward` 
        method should take a tensor of shape `(batch_size, forward_out+1)` and 
        return a tensor of shape `(batch_size, 1)`, where 
        `(batch_size, forward_out)` is the shape of the output from the 
        `self.forward` method when the batch size is `batch_size`; the "+1"
        in the input shape comes from that the energy also has the time as an
        input. If the `self.forward` method is not specified `forward_out` 
        is the size of the covariate/feature vector.

    References
    ----------

    [1] Holmer, Olov, Erik Frisk, and Mattias Krysander. "Energy-Based Survival
    Models for Predictive Maintenance." IFAC-PapersOnLine 56.2 (2023): 10862-10867.
    https://doi.org/10.48550/arXiv.2302.00629

    """

    def __init__(self, max_time:float,  
                 train_integrator: Union[EnergyIntegrator, int], 
                 eval_integrator: Union[EnergyIntegrator, int], 
                 tail_ratio:float=1.2, 
                 energy_net: Optional[Module]=None):
        
        super().__init__()

        self._max_time = max_time
        self.tail_ratio = tail_ratio

        if isinstance(train_integrator, EnergyIntegrator):
            self._train_integrator = train_integrator
        else:
            self._train_integrator = RandomMonteCarlo(int(train_integrator))

        if isinstance(eval_integrator, EnergyIntegrator):
            self._eval_integrator = eval_integrator
        else:
            self._eval_integrator = EquidistantTrapezoidal(int(eval_integrator))
            
        self._active_integrator = train_integrator

        if energy_net is not None:
            self._energy_net = energy_net
        else:
            self._energy_net = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._active_integrator = self._train_integrator
        else:
            self._active_integrator = self._eval_integrator
        return self

    def eval(self):
        return self.train(False)
    
    def max_time(self):
        return self._max_time

    def forward_output_size(self):
        return 1

    def _log_likelihood(self, forward_out, T, E):

        log_Z0 = self._active_integrator.log_integrate(
            self, forward_out, torch.zeros((T.shape[0], 1)), self._max_time, self.tail_ratio)

        log_Z = self._active_integrator.log_integrate(
            self, forward_out, T, self._max_time, self.tail_ratio)

        z = self.energy(torch.cat((forward_out, T.reshape(-1, 1)), dim=1))

        l = - log_Z0
        l[E] += z[E]
        l[~E] += log_Z[~E]
        return l

    def _density_function(self, forward_out, T):
        Z = self._active_integrator.integrate(
            self, forward_out, torch.zeros((T.shape[0], 1)), self._max_time, self.tail_ratio)

        f = torch.exp(self.energy(torch.cat((forward_out, T.reshape(-1, 1)), dim=1))) \
            / Z.reshape(-1, 1)
        return f

    def _survival_function(self, forward_out, T):
        Z0 = self._active_integrator.integrate(
            self, forward_out, torch.zeros((T.shape[0], 1)), self._max_time, self.tail_ratio)
        Z = self._active_integrator.integrate(
            self, forward_out, T, self._max_time, self.tail_ratio)
        return Z/Z0

    def energy(self, X_T):
        return self._energy_net(X_T)
    
    def forward(self, X):
        return X

    @classmethod
    def MLP_implementation(cls, max_time, train_integrator, eval_integrator,
                           input_size, hidden_sizes,
                           tail_ratio=1.2,
                           activation=torch.nn.ReLU, dropout=False,
                           batch_norm=False):

        return MLP_EBD(max_time,  train_integrator, eval_integrator,
                       input_size, hidden_sizes,tail_ratio,
                       activation, dropout,
                       batch_norm)


class MLP_EBD(EnergyBasedDensity):
    def __init__(self, max_time, train_integrator, eval_integrator,
                 input_size, hiden_sizes, tail_ratio=1.2,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        

        super(MLP_EBD, self).__init__(max_time, train_integrator, eval_integrator,
                                      tail_ratio)
        
        self.layers = MLP(input_size=input_size+1,
                          output_size=1,
                          hidden_sizes=hiden_sizes,
                          activation=activation,
                          dropout=dropout,
                          batch_norm=batch_norm)
        self.input_size = input_size

    def energy(self, X_T):
        return self.layers(X_T)

