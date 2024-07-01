import torch
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models.piecewise.grid import DiscretizationGrid
from pysare.models.utils import MLP
from abc import abstractmethod


def reverse_cumsum(x):
    cum_sum = torch.cumsum(x, dim=1)
    return x - cum_sum + cum_sum[:, -1].view(-1, 1)


class LinearDensity(TorchSurvivalModel):
    r"""Implements an piecewise linear density model.
    
    The forward output size should be 
        number of intervals + 2 + potential extra inputs required by the grid

    See :mod:`pysare.models.piecewise` for information on how to use 
    piecewise models.
    """
    def __init__(self, grid: DiscretizationGrid):
        super(LinearDensity, self).__init__()
        self.grid = grid

    def _bin_and_normalize(self, z, T):
        # Number of subjects
        N = T.shape[0]
        # Binwidths


        disc, z = self.grid.bin(T, z)

        e = torch.exp(z)

        # Calculate S times Z at each tau
        # dF = e[:, :-1]*disc.dt

        dF = e[:, 1:].clone()
        dF[:, :-1] += e[:, :-2]
        dF[:, :-1] *= disc.dt/2

        SZ = reverse_cumsum(dF)

        # Normalization constant is first collumn in S*Z (1*Z)
        Z = SZ[:, 0]

        e_ind = e[range(N), disc.ind]
        e_next_ind = e[range(N), disc.ind+1]

        return SZ, Z, z, e, disc, e_ind, e_next_ind

    def forward_output_size(self):
        return self.grid.num_intervals()+self.grid.extra_forward_output_size()+2

    def max_time(self):
        return self.grid.max_time()

    def _log_likelihood(self, z, T, E):
        N = T.shape[0]

        SZ, Z, z, e, disc, e_ind, e_next_ind = self._bin_and_normalize(z, T)

        log_likelihood = -torch.log(Z)

        ind_f = E
        ind_c = ~E

        log_likelihood[ind_f] += torch.log(e[ind_f, disc.ind[ind_f]] + (T[ind_f]-disc.tau_ind[ind_f]) * (
            e[ind_f, disc.ind[ind_f]+1]-e[ind_f, disc.ind[ind_f]])/disc.dt_ind.expand(N)[ind_f])
        log_likelihood[ind_c] += torch.log(SZ[ind_c, disc.ind[ind_c]+1] + e_next_ind[ind_c]*(disc.tau_next_ind[ind_c] - T[ind_c]) + (
            (disc.tau_next_ind[ind_c] - T[ind_c])**2)/(2*disc.dt_ind.expand(N)[ind_c])*(e_ind[ind_c] - e_next_ind[ind_c]))

        return log_likelihood

    def _density_function(self, z, T):
        SZ, Z, z, e, disc, e_ind, e_next_ind = self._bin_and_normalize(z, T)

        N = T.shape[0]
        f = (e[range(N), disc.ind] + (T-disc.tau_ind) *
             (e[range(N), disc.ind+1]-e[range(N), disc.ind])/disc.dt_ind)/Z
        return f

    def _survival_function(self, z, T):
        SZ, Z, z, e, disc, e_ind, e_next_ind = self._bin_and_normalize(z, T)
        N = T.shape[0]

        S = (SZ[range(N), disc.ind+1] + e_next_ind*(disc.tau_next_ind - T) +
             ((disc.tau_next_ind - T)**2)/(2*disc.dt_ind)*(e_ind - e_next_ind))/Z
        return S

    def forward(self, X):
        return X

    @classmethod
    def MLP_implementation(cls, grid, input_size, hidden_sizes,
                           activation=torch.nn.ReLU, dropout=False,
                           batch_norm=False):
        return MLPLinearDensity(grid, input_size, hidden_sizes,
                                activation=torch.nn.ReLU, dropout=0.0,
                                batch_norm=False)

class MLPLinearDensity(LinearDensity):
    def __init__(self, grid, input_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        super(MLPLinearDensity, self).__init__(grid=grid)

        self.input_size = input_size

        self.layers = MLP(input_size, grid.num_intervals() + grid.extra_forward_output_size() + 2,
                          hidden_sizes, activation, dropout, batch_norm)

    def forward(self, X):
        logits = self.layers(X)

        return logits

