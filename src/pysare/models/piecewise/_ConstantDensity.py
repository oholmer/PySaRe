import torch
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models.piecewise.grid import DiscretizationGrid
from pysare.models.utils.MLP import MLP


def reverse_cumsum(x):
    cum_sum = torch.cumsum(x, dim=1)
    return x - cum_sum + cum_sum[:, -1].view(-1, 1)


class ConstantDensity(TorchSurvivalModel):
    r"""Implements an piecewise constant density model
    
    The forward output size should be 
        number of intervals + 1 + potential extra inputs required by the grid

    See :mod:`pysare.models.piecewise` for information on how to use 
    piecewise models.
    """

    def __init__(self, grid: DiscretizationGrid):
        super(ConstantDensity, self).__init__()
        self.grid = grid

    def _bin_and_normalize(self, z, T):

        # Number of subjects
        N = T.shape[0]
        # Binwidths

        disc, z = self.grid.bin(T, z)

        e = torch.exp(z)

        # Calculate S times Z at each tau
        dF = e.clone()
        dF[:, :-1] *= disc.dt
        SZ = reverse_cumsum(dF)

        # Normalization constant is first collumn in S*Z (1*Z)
        Z = SZ[:, 0]

        return SZ, Z, z, e, disc.ind, disc.tau_next_ind

    def forward_output_size(self):
        return self.grid.num_intervals()+self.grid.extra_forward_output_size()+1

    def max_time(self):
        return self.grid.max_time()

    def _log_likelihood(self, z, T, E):

        SZ, Z, z, e, ind, tau_next = self._bin_and_normalize(z, T)

        log_likelihood = -torch.log(Z)

        ind_f = torch.where(E)[0]
        ind_c = torch.where(~E)[0]
        log_likelihood[ind_f] += z[ind_f, ind[ind_f]]
        log_likelihood[ind_c] += torch.log(SZ[ind_c, ind[ind_c]+1] +
                                           (tau_next[ind_c]-T[ind_c])*e[ind_c, ind[ind_c]])

        return log_likelihood

    def _density_function(self, z, T):
        SZ, Z, z, e, ind, tau_next = self._bin_and_normalize(z, T)
        N = T.shape[0]
        f = e[range(N), ind]/Z
        return f

    def _survival_function(self, z, T):
        SZ, Z, z, e, ind, tau_next = self._bin_and_normalize(z, T)
        N = T.shape[0]
        S = (SZ[range(N), ind+1] + (tau_next-T)*e[range(N), ind])/Z
        return S

    def forward(self, X):
        return X

    @classmethod
    def MLP_implementation(cls, grid, input_size, hidden_sizes,
                           activation=torch.nn.ReLU, dropout=False,
                           batch_norm=False):
        return MLPConstantDensity(grid, input_size, hidden_sizes,
                                  activation, dropout,
                                  batch_norm)


class MLPConstantDensity(ConstantDensity):
    def __init__(self, grid, input_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        super(MLPConstantDensity, self).__init__(grid=grid)

        self.input_size = input_size

        self.layers = MLP(input_size, grid.num_intervals() + grid.extra_forward_output_size() + 1,
                          hidden_sizes, activation, dropout, batch_norm)

    def forward(self, X):
        return self.layers(X)
