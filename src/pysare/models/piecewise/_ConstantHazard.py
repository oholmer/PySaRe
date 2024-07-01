import torch
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models.piecewise.grid import DiscretizationGrid
from pysare.models.utils.MLP import MLP
from abc import abstractmethod


class ConstantHazard(TorchSurvivalModel):
    r"""Implements an piecewise constant hazard model.
    
    The forward output size should be 
        number of intervals + potential extra inputs required by the grid

    See :mod:`pysare.models.piecewise` for information on how to use 
    piecewise models.
    """
    def __init__(self, grid: DiscretizationGrid):
        super(ConstantHazard, self).__init__()
        self.grid = grid

    def _log_likelihood(self, z, T, E):
        disc, z = self.grid.bin(T, z)
        N = T.shape[0]

        e = torch.exp(z)

        ln_e = z

        dH = e*disc.dt
        H_n = dH.cumsum(dim=1)
        h = e[range(N), disc.ind]
        H = H_n[range(N), disc.ind] - (disc.tau_next_ind-T)*h
        # S = torch.exp(-H)
        # f = h*S

        ind_f = torch.where(E)[0]
        ind_c = torch.where(~E)[0]
        ln_f = ln_e[range(N), disc.ind] - H
        ln_S = -H
        log_likelihood = ln_f
        log_likelihood[ind_c] = ln_S[ind_c]

        return log_likelihood

    def max_time(self):
        return self.grid.max_time()
    
    def forward_output_size(self):
        return self.grid.num_intervals()+self.grid.extra_forward_output_size()

    def _density_function(self, z, T):
        N = T.shape[0]
        disc, z = self.grid.bin(T, z)

        e = torch.exp(z)

        

        dH = e*disc.dt
        H_n = dH.cumsum(dim=1)
        h = e[range(N), disc.ind]
        H = H_n[range(N), disc.ind] - (disc.tau_next_ind-T)*h
        S = torch.exp(-H)
        # f = h*S
        return h*S

    def _survival_function(self, z, T):
        N = T.shape[0]
        disc, z = self.grid.bin(T, z)
        e = torch.exp(z)

        

        dH = e*disc.dt
        H_n = dH.cumsum(dim=1)
        h = e[range(N), disc.ind]
        H = H_n[range(N), disc.ind] - (disc.tau_next_ind-T)*h
        S = torch.exp(-H)
        return S

    def forward(self, X):
        return X

    @classmethod
    def MLP_implementation(cls, grid, input_size, hidden_sizes,
                           activation=torch.nn.ReLU, dropout=False,
                           batch_norm=False):
        return MLPConstantHazard(grid, input_size, hidden_sizes,
                                 activation, dropout,
                                 batch_norm)


class MLPConstantHazard(ConstantHazard):
    def __init__(self, grid, input_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        super(MLPConstantHazard, self).__init__(grid=grid)

        self.input_size = input_size

        self.layers = MLP(input_size, grid.num_intervals() + grid.extra_forward_output_size(),
                          hidden_sizes, activation, dropout, batch_norm)

    def forward(self, X):
        return self.layers(X)
