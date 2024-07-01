from torch.nn import Module
from torch import Tensor
import torch
from ._numpy.NumpySurvivalModel import NumpySurvivalModel

from typing import Optional, Union, Iterable, Callable, Literal, TypeVar, NewType, Any, Tuple
from pysare._typing import X, T, E
from pysare.data import Dataset
from torch.utils.data import DataLoader


class TorchSurvivalModel(Module):
    r"""Base class for survival models.

    This class is subclassed in all torch-based survival models.
    """

    def __init__(self):
        super(TorchSurvivalModel, self).__init__()

    def max_time(self) -> float:
        r"""Returns the maximal time for which the model is defined"""
        raise NotImplementedError()

    def forward_output_size(self, *args, **kwargs):
        r"""Returns the required size of the forward output."""
        raise NotImplementedError()

    def to_numpy(self,
                 data: Optional[Union[Tuple[X, T, E],
                                      Dataset,
                                      DataLoader]] = None,
                 torch_max_time: Optional[float] = None,
                 numpy_support: Optional[Tuple[float, float]] = None,
                 batch_size: Optional[int] = None,
                 extrapolation: Literal['set_zero', 'model'] = 'set_zero') -> NumpySurvivalModel:
        r"""Converts the model to a NumpySurvivalModel that accepts numpy inputs.

        Implements an interface to a PyTorch model creating a survival model that
        takes NumPy arrays (or any object that can be converted to such) as input.


        Parameters
        ----------

        data : tuple(X, T, E) or Dataset or DataLoader, optional
            Data used to initialize the interface. If data should represent the
            tensors X, T, and E, and the device and dtype of these tensors
            are used to determine the correct tensor types for the torch_model.
            If data is a DataLoader its batch size will also be used in the
            interface, if not batch_size is set.
        torch_max_time : float, optional
            Defines the maximal time of the original model before extrapolation 
            is to be used. By default, self.max_time() is used.
        numpy_support : tuple(float, float), optional
            Support of the constructed NumPy model. The numpy model will map
            [0.0, torch_max_time] onto numpy_support using an affine
            transformation. By default the same support as the original model 
            is used.
        batch_size : int, optional
            Maximal batch size to be used when calling the underlying PyTorch
            Model. By default an infinite batch size, or the size of the
            DataLoader optionally specified as the data parameter.
        extrapolation : {'set_zero', 'model'}, default='set_zero'
            Determines how the model is extrapolated beyond torch_max_time.
            - 'set_zero' -- Survival function is set to zero
            - 'model' -- The value given by the torch_model is used even beyond
                its maximal time. Can result in unexpected behavior and should be
                used with caution.
        """
        return NumpySurvivalModel(self,
                                  data=data,
                                  torch_max_time=torch_max_time,
                                  numpy_support=numpy_support,
                                  batch_size=batch_size,
                                  extrapolation=extrapolation)

    def sample(self, X: Tensor, num_samples: int = 1, generator: Optional[torch.Generator] = None, **kwargs):
        with torch.no_grad():
            return self._sample(self.forward(X), num_samples, generator, **kwargs)

    def _sample(self, forward_out, num_samples=1, generator=None, burn_in=20, shuffle=True):
        r"""Metropolis-Hastings sampling

        Default sampling method for models that does not have a specific 
        implementation.

        The Metropolis-Hastings algorithm is used to sample from the
        distribution of (T | T<max_time). After the Metropolis-Hastings 
        algorithm, each sample is set to max_time with probability
        1-S(max_time).

        """

        N = forward_out.shape[0]

        if not self.max_time() < torch.inf:
            raise NotImplementedError("This model has nonfinite max time" +
                                      "and can therefore not be sampled " +
                                      "using Metropolis-Hastings ")
        else:
            max_time = self.max_time()
            S_max_time = self._survival_function(
                forward_out,
                torch.full((N,),
                           max_time,
                           device=forward_out.device,
                           dtype=forward_out.dtype))

        T = torch.empty((N, num_samples),
                        device=forward_out.device,
                        dtype=forward_out.dtype)

        T_last = torch.rand((N,),
                            device=forward_out.device,
                            dtype=forward_out.dtype,
                            generator=generator)*max_time

        if burn_in < 1:
            T[:, 0] = T_last
            # Tail of the distribution
            T[torch.rand(T_last.shape, dtype=T_last.dtype,
                         device=T_last.device, generator=generator)
              < S_max_time, 0] = max_time

        f_last = self._density_function(forward_out, T_last)

        for n in range(1, num_samples+burn_in):
            proposal = torch.rand((N,),
                                  device=forward_out.device,
                                  dtype=forward_out.dtype,
                                  generator=generator)*max_time

            f_new = self._density_function(forward_out, proposal)

            accept = torch.rand((N,),
                                device=forward_out.device,
                                dtype=forward_out.dtype,
                                generator=generator) <= f_new/f_last

            if accept.any():
                T_last[accept] = proposal[accept]
                f_last[accept] = f_new[accept]

            if n >= burn_in:
                T[:, n-burn_in] = T_last
                # Tail of the distribution
                T[torch.rand(T_last.shape, dtype=T_last.dtype,
                             device=T_last.device, generator=generator)
                  < S_max_time, n-burn_in] = max_time

        if shuffle:
            shuffling_inds = torch.argsort(torch.rand(T.shape,
                                                      dtype=T.dtype,
                                                      device=T.device,
                                                      generator=generator),
                                           dim=1)
            return torch.gather(T, dim=1, index=shuffling_inds)
        else:
            return T

    def log_likelihood(self, X: Tensor, T: Tensor, E: Tensor) -> Tensor:
        r"""Returns the likelihood of the observation (X,T,E)

        Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times
            E : (n,) tensor
                Tensor with event types
        """
        Z = self.forward(X)
        return self._log_likelihood(Z, T, E)

    def _log_likelihood(self, Z: Tensor, T: Tensor, E: Tensor) -> Tensor:

        log_likelihood = torch.empty_like(T)

        if torch.any(E):
            log_likelihood[E] = self._log_density_function(Z[E], T[E])

        if torch.any(~E):
            log_likelihood[~E] = self._log_survival_function(Z[~E], T[~E])

        return log_likelihood

    def density_function(self, X: Tensor, T: Tensor) -> Tensor:
        r"""Lifetime density at T given X

        Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times

        """
        return self._density_function(self.forward(X), T)

    def _density_function(self, Z: Tensor, T: Tensor) -> Tensor:
        raise NotImplementedError()

    def hazard_function(self, X: Tensor, T: Tensor) -> Tensor:
        r"""Hazard function at at T given X

           Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times

        """
        return self._hazard_function(self.forward(X), T)

    def _hazard_function(self, Z: Tensor, T: Tensor) -> Tensor:
        return self._density_function(Z, T)/self._survival_function(Z, T)

    def log_density_function(self, X: Tensor, T: Tensor) -> Tensor:
        r"""Logarithm of the density function at T given X

         Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times

        """
        return self._log_density_function(self.forward(X), T)

    def _log_density_function(self, Z: Tensor, T: Tensor) -> Tensor:
        return torch.log(self._density_function(Z, T))

    def survival_function(self, X: Tensor, T: Tensor) -> Tensor:
        r"""Survival function at T given X

         Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times

        """
        return self._survival_function(self.forward(X), T)

    def _survival_function(self, Z: Tensor, T: Tensor) -> Tensor:
        raise NotImplementedError()

    def log_survival_function(self, X: Tensor, T: Tensor) -> Tensor:
        r"""Logarithm of the survival function at T given X

         Parameters
        ----------

            X : (n, ...) tensor
                Tensor with features/covariate with first dimension as batch 
                dimension 
            T : (n, ) tensor
                Tensor with times
        """
        return self._log_survival_function(self.forward(X), T)

    def _log_survival_function(self, Z: Tensor, T: Tensor) -> Tensor:
        return torch.log(self._survival_function(Z, T))


class SplineSurvivalModel(TorchSurvivalModel):
    def __init__(self, times, probabilities):
        super(SplineSurvivalModel, self).__init__()

        self.times = torch.nn.Parameter(torch.cat((times, torch.tensor(
            [2*times[-1]-times[-2]], dtype=times.dtype, device=times.device)), dim=-1), requires_grad=False)
        self.probabilities = torch.nn.Parameter(torch.cat((probabilities, torch.tensor(
            [0.0], dtype=probabilities.dtype, device=probabilities.device)), dim=-1), requires_grad=False)
        self.dt = torch.nn.Parameter(self.times.diff(), requires_grad=False)
        self.density = torch.nn.Parameter(-self.probabilities.diff() /
                                          self.dt, requires_grad=False)

    def sample(self, X, generator=None):

        u = torch.rand(X.shape[0], generator=generator)

        # Search sorted requires an increasing sequence, so CDF probabilities
        # are used
        P = 1-self.probabilities
        ind = torch.searchsorted(P, u, right=True)-1
        ind[u == 1-self.probabilities[-1]] -= 1

        time = self.times[ind]
        density = self.density[ind]
        probability = P[ind]

        return (time + (u-probability)/density).clip(0, self.max_time())

    def max_time(self):
        return self.times[-2]

    def _find_ind(self, T):
        T = T.clip(0, self.times[-1])
        ind = torch.searchsorted(self.times, T, right=True)-1
        ind[T == self.times[-1]] = len(self.times)-2
        return ind

    def _density_function(self, X, T, batch_size=None) -> Tensor:
        return self.density[self._find_ind(T)]

    def _survival_function(self, X, T, batch_size=None) -> Tensor:
        ind = self._find_ind(T)

        density = self.density[ind]
        probability = self.probabilities[ind]

        dt = self.dt[ind]
        time = self.times[ind]

        return probability-density*(T-time)

    def to_numpy(self, dtype=None, device=None):
        return NumpySurvivalModel(self, dtype, device, )


class FrozenSurvivalModel(Module):
    def __init__(self, base_model, X):
        super(FrozenSurvivalModel, self).__init__()

        self.base_model = base_model
        self.X = X

    def _feature_vector(self, N):
        return self.X.expand(N, *self.X.shape)

    def max_time(self):
        return self.base_model.max_time()

    def sample(self, size=1, generator=None):
        return self.base_model.sample(self._feature_vector(size), generator)

    def log_likelihood(self, T, E, batch_size=None) -> Tensor:
        return self.base_model.log_likelihood(self._feature_vector(len(T)), T.reshape(-1,), E.reshape(-1,))

    def lifetime_density(self, T, batch_size=None) -> Tensor:
        return self.base_model.density_function(self._feature_vector(len(T)), T.reshape(-1,))

    def survival_function(self, T, batch_size=None) -> Tensor:

        return self.base_model.survival_function(self._feature_vector(len(T)), T.reshape(-1,))

    def to_numpy(self, dtype=None, device=None):
        return NumpyFrozenSurvivalModel(self, dtype, device)

    def to_spline_approximation(self, discretization):
        if torch.is_tensor(discretization) and (discretization.numel() > 1):
            times = discretization
        else:
            times = torch.linspace(
                0, self.max_time(), int(discretization), dtype=self.X.dtype, device=self.X.device)

        probabilities = self.survival_function(times)

        return FrozenSurvivalModel(SplineSurvivalModel(times, probabilities), self.X)
