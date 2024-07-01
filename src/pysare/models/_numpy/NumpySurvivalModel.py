from __future__ import annotations
import numpy as np
import torch
import warnings
from collections.abc import Iterable
import warnings
from pysare._typing import X, T, E
from typing import Optional, Union, Iterable, Callable, Literal, TypeVar, NewType, Any, List
from typing import Tuple
from torch import Tensor
from torch.nn import Module
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.sampler
import numpy as np
from numpy.typing import NDArray

# from pysare.models.SurvivalModel import TorchSurvivalModel


def try_find_parameter_type(torch_model):
    # Loop through parameters until a floating point
    for parameter in torch_model.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype, parameter.device

    # If model is a distribution no warning is given
    from pysare.models.distributions import _distributions
    if not torch_model.__class__ in _distributions:
        warnings.warn(
            "Could not determine device and dtype from self.parameters()," +
            "using the default tensor type")

    # Returing default tensor type
    return torch.empty(0).dtype, torch.empty(0).device


def convert_to_tensor(data, dtype, device):
    if torch.is_tensor(data):
        return data.to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


INDEX_DTYPE = torch.long


def validate_inputs(batch_size, max_time, X, T, E=None):

    # If E is provided it must have same shape as T
    if (not E is None) and (not E.shape == T.shape):
        raise ValueError("The shape of T and E must be equal")

    if len(X.shape) < 2:
        raise ValueError(
            "The number of dimensions of X must be larger than 1.")

    N = X.shape[0]

    squeeze = False

    if len(T.shape) > 2:
        raise ValueError(
            "The number of dimensions of T must be either 1 or 2.")

    elif len(T.shape) == 0:
        # T is scalar
        # ----------------------------------------------------------------------

        T_inds = torch.full((N,), 0, dtype=INDEX_DTYPE, device=T.device)
        X_inds = torch.arange(0, N, 1, dtype=INDEX_DTYPE, device=X.device)
        T_out_inds = T_inds.cpu()
        X_out_inds = X_inds.cpu()

        if N > 1:
            output_shape = (N, 1)
        else:
            output_shape = (1, 1)
            squeeze = True

        T = T.reshape(-1,)

    elif len(T.shape) == 1:
        # T has shape (M,)
        # ----------------------------------------------------------------------
        M = T.shape[0]
        T_inds = torch.arange(0, M, 1, dtype=INDEX_DTYPE,
                              device=T.device).repeat(N)
        X_inds = torch.arange(0, N, dtype=INDEX_DTYPE,
                              device=X.device).repeat_interleave(M)
        T_out_inds = T_inds.cpu()
        X_out_inds = X_inds.cpu()
        if N == 1:
            output_shape = (1, M)
            squeeze = True
        else:
            output_shape = (N, M)

    elif T.shape[0] == 1:
        # T has shape (1,M)
        # ----------------------------------------------------------------------
        M = T.shape[1]
        T_inds = torch.arange(0, M, 1, dtype=INDEX_DTYPE,
                              device=T.device).repeat(N)
        X_inds = torch.arange(0, N, dtype=INDEX_DTYPE,
                              device=X.device).repeat_interleave(M)
        T_out_inds = T_inds.cpu()
        X_out_inds = X_inds.cpu()
        output_shape = (N, M)

    elif T.shape[0] == X.shape[0]:
        if T.shape[1] != 1:
            raise ValueError("Either T.shape[0] or T.shape[1] must be 1.")
        # T has shape (N,1)
        # ----------------------------------------------------------------------
        # T is a column vector for witch row i is applied to row i in X
        T_inds = torch.arange(0, N, 1, dtype=INDEX_DTYPE, device=T.device)
        X_inds = torch.arange(0, N, 1, dtype=INDEX_DTYPE, device=X.device)
        T_out_inds = torch.full((N,), 0, dtype=INDEX_DTYPE, device='cpu')
        X_out_inds = X_inds.cpu()

        output_shape = (N, 1)
    else:
        raise ValueError(
            "T.shape[0] must be equal to 1 or equal to X.shape[0].")

    left_inds = T[T_inds].reshape(-1,) < 0
    right_inds = T[T_inds].reshape(-1,) > max_time
    mid_inds = ~left_inds & ~right_inds

    if batch_size is None:
        batch_size = len(X_inds)

    # return X_inds, X_out_inds, T_inds, T_out_inds
    return ([[data[inds].split(batch_size)
             for inds in (left_inds, mid_inds, right_inds)]
            for data in (X_inds, X_out_inds, T_inds, T_out_inds)],
            output_shape,
            squeeze,
            batch_size)


class NumpySurvivalModel():
    r"""NumPy interface to PyTorch Model

    Implements an interface to a PyTorch model creating a survival model that
    takes NumPy arrays (or any object that can be converted to such) as input.

    Typically created by calling the method `to_numpy` on a PyTorch Model.

    Parameters
    ----------

    torch_model : TorchSurvivalModel
        The original PyTorch model to be interfaced
    torch_max_time : float, optional
        Maximal time for which the original PyTorch model is defined. The
        By default, the support torch_model.max_time() is used.
    numpy_support : tuple(float, float), optional
        Support of the constructed NumPy model. The numpy model will map
        [0.0, torch_max_time] onto numpy_support using an affine
        transformation. By default the same support as the original PyTorch
        model is used.
    batch_size : int, optional
        Maximal batch size to be used when calling the underlying PyTorch
        Model. By default an infinite batch size, or the size of the
        DataLoader optionally specified as the data parameter.
    data : tuple(X, T, E) or Dataset or DataLoader, optional
        Data used to initialize the interface. If data should represent the
        tensors X, T, and E, and the device and dtype of these tensors
        are used to determine the correct tensor types for the torch_model.
        If data is a DataLoader its batch size will also be used in the
        interface, if not batch_size is set.
    extrapolation : {'set_zero', 'model'}, default='set_zero'
        Determines how the model is extrapolated beyond torch_max_time.
        - 'set_zero' -- Survival function is set to zero
        - 'model' -- The value given by the torch_model is used even beyond
            its maximal time. Can result in unexpected behavior and should be
            used with caution.
    """

    def __init__(self,
                 torch_model: Any,
                 torch_max_time: Optional[float] = None,
                 numpy_support: Optional[Tuple[float, float]] = None,
                 batch_size: Optional[int] = None,
                 data: Optional[Union[Tuple,
                                      pysare.data.Dataset,
                                      torch.utils.data.DataLoader]] = None,
                 extrapolation: Literal['set_zero', 'model'] = 'set_zero'):

        if data is None:

            dtype, device = try_find_parameter_type(torch_model)

            def data_transform(X, T, E):
                return (convert_to_tensor(X, dtype=dtype, device=device),
                        convert_to_tensor(T, dtype=dtype, device=device),
                        convert_to_tensor(E, dtype=torch.bool, device=device))
        else:
            if isinstance(data, Iterable):
                temp = next(iter(data))
                if isinstance(temp, Union[List,Tuple]):
                    # data is a data loader/iterates over batches
                    X, T, E = temp
                else:
                    # data is a tuple (X, T, E)
                    X, T, E = data
            else:
                # Data is Dataset
                X, T, E = data[:]

            X_device = X.device
            X_dtype = X.dtype
            T_device = T.device
            T_dtype = T.dtype
            E_device = E.device

            def data_transform(X, T, E):
                return (convert_to_tensor(X,
                                          dtype=X_dtype,
                                          device=X_device),
                        convert_to_tensor(T,
                                          dtype=T_dtype,
                                          device=T_device),
                        convert_to_tensor(E,
                                          dtype=torch.bool,
                                          device=E_device))

        self.data_transform = data_transform
        self.torch_model = torch_model

        # Batch size
        if (batch_size is None) and hasattr(data, 'batch_size'):
            self.batch_size = data.batch_size
        else:
            self.batch_size = batch_size

        # Support
        if torch_max_time is None:
            self.torch_max_time = torch_model.max_time()
        else:
            self.torch_max_time = torch_max_time
        if numpy_support is None:
            self.numpy_support = (0, self.torch_max_time)
        else:
            if not numpy_support[1] < np.inf:
                raise ValueError("Can not transform support of model with" +
                                 "infinite support.")
            self.numpy_support = tuple(numpy_support)

        self.numpy_min_time = self.numpy_support[0]
        if self.torch_max_time < np.inf:
            self.numpy_scale = self.torch_max_time/(self.numpy_support[1]
                                                    - self.numpy_support[0])
        else:
            self.numpy_scale = 1.0

        self.extrapolation = extrapolation

    def freeze(self, X: Any) -> FrozenNumpySurvivalModel:
        r"""Freezes the model around X,

        Freezes the model around a set of covariates/features represented by
        the ArrayLike parameter X"""
        with torch.no_grad():
            self.torch_model.eval()
            X, _, _ = self.data_transform(X, torch.empty(0), torch.empty(0))
            if self.batch_size is None:
                batch_size = X.shape[0]
            else:
                batch_size = self.batch_size

            forward_out = torch.concat([self.torch_model.forward(X_batch)
                                        for X_batch in X.split(batch_size, dim=0)],
                                       dim=0)

            return FrozenNumpySurvivalModel(self, forward_out)

    def to_torch(self) -> TorchSurvivalModel:
        r"""Returns the underlying PyTorch Model."""
        return self.torch_model

    def max_time(self) -> float:
        r"""Returns the maximal time after which extrapolation is used."""
        return self.numpy_support[1]

    def log_likelihood(self, X: Any, T: Any, E: Any) -> NDArray:
        r"""Returns the log-likelihood given X, T, and E.

        Parameters
        ----------

        X : (N,...) array_like
            Array with features/covariates with at least two dimensions,
            where the first dimension is the batch dimension of size N.
        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)

        E : (N, 1) or (N, 1) or (N,) bool array_like
            Array of event types with the same size as T. Entries evaluated
            as True indicates recorded events and entries evaluated as False
            indicate censored times"""

        self.torch_model.eval()
        with torch.no_grad():

            X, T, E = self.data_transform(X, T, E)

            T = (T-self.numpy_min_time)*self.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, batch_size) = validate_inputs(self.batch_size,
                                                                 self.torch_max_time,
                                                                 X,
                                                                 T)

            T = T.reshape(-1,)
            E = E.reshape(-1,)

            forward_out = torch.concat([self.torch_model.forward(X_batch)
                                        for X_batch in X.split(batch_size, dim=0)],
                                       dim=0)

            return self._log_likelihood(forward_out, T, E,
                                        X_ind_batches, X_out_ind_batches,
                                        T_ind_batches, T_out_ind_batches,
                                        output_size, squeeze)

    def _log_likelihood(self, forward_out, T, E,
                        X_ind_batches, X_out_ind_batches,
                        T_ind_batches, T_out_ind_batches,
                        output_size, squeeze):

        log_lik = np.full(output_size, np.NINF)

        # T<0 inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[0],
                                                          X_out_ind_batches[0],
                                                          T_ind_batches[0],
                                                          T_out_ind_batches[0]):

            inds = ~E[T_inds]
            if len(inds):
                log_lik[X_out_inds[inds], T_out_inds[inds]] = np.log(1.0)

        # T in support inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[1],
                                                          X_out_ind_batches[1],
                                                          T_ind_batches[1],
                                                          T_out_ind_batches[1]):
            log_lik[X_out_inds, T_out_inds] = self.torch_model._log_likelihood(
                forward_out[X_inds], T[T_inds], E[T_inds]).numpy()

        # T>max_time inds
        if self.extrapolation == 'model':
            for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[2],
                                                              X_out_ind_batches[2],
                                                              T_ind_batches[2],
                                                              T_out_ind_batches[2]):
                log_lik[X_out_inds, T_out_inds] = self.torch_model._log_likelihood(
                    forward_out[X_inds], T[T_inds], E[T_inds]).numpy()

        # Add shape constant to events
        log_lik[E.cpu()] += np.log(self.numpy_scale)

        if squeeze:
            return log_lik.squeeze()
        else:
            return log_lik

    def density_function(self, X: Any, T: Any) -> NDArray:
        r"""Returns the density at T given X.

        Parameters
        ----------

        X : (N,...) array_like
            Array with features/covariates with at least two dimensions,
            where the first dimension is the batch dimension of size N.
        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """
        self.torch_model.eval()
        with torch.no_grad():

            X, T, _ = self.data_transform(X, T, torch.empty(0))

            T = (T-self.numpy_min_time)*self.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, batch_size) = validate_inputs(self.batch_size,
                                                                 self.torch_max_time,
                                                                 X,
                                                                 T)

            T = T.reshape(-1,)

            forward_out = torch.concat([self.torch_model.forward(X_batch)
                                        for X_batch in X.split(batch_size, dim=0)],
                                       dim=0)

            return self._density_function(forward_out, T,
                                          X_ind_batches, X_out_ind_batches,
                                          T_ind_batches, T_out_ind_batches,
                                          output_size, squeeze)

    def _density_function(self, forward_out, T,
                          X_ind_batches, X_out_ind_batches,
                          T_ind_batches, T_out_ind_batches,
                          output_size, squeeze):

        f = np.zeros(output_size)

        # T in support inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[1],
                                                          X_out_ind_batches[1],
                                                          T_ind_batches[1],
                                                          T_out_ind_batches[1]):
            f[X_out_inds, T_out_inds] = self.torch_model._density_function(
                forward_out[X_inds], T[T_inds]).numpy().squeeze()*self.numpy_scale

        # T>max_time inds
        if self.extrapolation == 'model':
            for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[2],
                                                              X_out_ind_batches[2],
                                                              T_ind_batches[2],
                                                              T_out_ind_batches[2]):
                f[X_out_inds, T_out_inds] = self.torch_model._density_function(
                    forward_out[X_inds], T[T_inds]).numpy().squeeze()*self.numpy_scale

        if squeeze:
            return f.squeeze()
        else:
            return f

    def survival_function(self, X, T) -> NDArray:
        r"""Returns the survival function at T given X.

        Parameters
        ----------

        X : (N,...) array_like
            Array with features/covariates with at least two dimensions,
            where the first dimension is the batch dimension of size N.
        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """
        self.torch_model.eval()
        with torch.no_grad():
            X, T, _ = self.data_transform(X, T, torch.empty(0))

            T = (T-self.numpy_min_time)*self.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, batch_size) = validate_inputs(self.batch_size,
                                                                 self.torch_max_time,
                                                                 X,
                                                                 T)

            T = T.reshape(-1,)

            forward_out = torch.concat([self.torch_model.forward(X_batch)
                                        for X_batch in X.split(batch_size, dim=0)],
                                       dim=0)
            return self._survival_function(forward_out, T,
                                           X_ind_batches, X_out_ind_batches,
                                           T_ind_batches, T_out_ind_batches,
                                           output_size, squeeze)

    def _survival_function(self, forward_out, T,
                           X_ind_batches, X_out_ind_batches,
                           T_ind_batches, T_out_ind_batches,
                           output_size, squeeze):
        S = np.zeros(output_size)
        # T<0 inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[0],
                                                          X_out_ind_batches[0],
                                                          T_ind_batches[0],
                                                          T_out_ind_batches[0]):
            S[X_out_inds, T_out_inds] = 1.0

        # T in support inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[1],
                                                          X_out_ind_batches[1],
                                                          T_ind_batches[1],
                                                          T_out_ind_batches[1]):
            S[X_out_inds, T_out_inds] = self.torch_model._survival_function(
                forward_out[X_inds], T[T_inds]).numpy().squeeze()

        # T>max_time inds
        if self.extrapolation == 'model':
            for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[2],
                                                              X_out_ind_batches[2],
                                                              T_ind_batches[2],
                                                              T_out_ind_batches[2]):
                S[X_out_inds, T_out_inds] = self.torch_model._survival_function(
                    forward_out[X_inds], T[T_inds]).numpy().squeeze()

        if squeeze:
            return S.squeeze()
        else:
            return S

    def hazard_function(self, X: Any, T: Any) -> NDArray:
        r"""Returns the hazard at T given X.

        Parameters
        ----------

        X : (N,...) array_like
            Array with features/covariates with at least two dimensions,
            where the first dimension is the batch dimension of size N.
        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """

        self.torch_model.eval()
        with torch.no_grad():

            X, T, _ = self.data_transform(X, T, torch.empty(0))

            T = (T-self.numpy_min_time)*self.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, batch_size) = validate_inputs(self.batch_size,
                                                                 self.torch_max_time,
                                                                 X,
                                                                 T)

            T = T.reshape(-1,)

            forward_out = torch.concat([self.torch_model.forward(X_batch)
                                        for X_batch in X.split(batch_size, dim=0)],
                                       dim=0)

            return self._hazard_function(forward_out, T,
                                         X_ind_batches, X_out_ind_batches,
                                         T_ind_batches, T_out_ind_batches,
                                         output_size, squeeze)

    def _hazard_function(self, forward_out, T,
                         X_ind_batches, X_out_ind_batches,
                         T_ind_batches, T_out_ind_batches,
                         output_size, squeeze):

        h = np.zeros(output_size)

        # T in support inds
        for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[1],
                                                          X_out_ind_batches[1],
                                                          T_ind_batches[1],
                                                          T_out_ind_batches[1]):
            h[X_out_inds, T_out_inds] = self.torch_model._hazard_function(
                forward_out[X_inds], T[T_inds]).numpy().squeeze()*self.numpy_scale

        # T>max_time inds
        if self.extrapolation == 'model':
            for X_inds, X_out_inds, T_inds, T_out_inds in zip(X_ind_batches[2],
                                                              X_out_ind_batches[2],
                                                              T_ind_batches[2],
                                                              T_out_ind_batches[2]):
                h[X_out_inds, T_out_inds] = self.torch_model._hazard_function(
                    forward_out[X_inds], T[T_inds]).numpy().squeeze()*self.numpy_scale

        if squeeze:
            return h.squeeze()
        else:
            return h

    def sample(self, X: Any, num_samples: int = 1,
               generator: Optional[torch.Generator] = None,
               **kwargs):
        r"""Generate samples from the distribution

        Parameters
        ----------

        X : (N,...) array_like
            Array with features/covariates with at least two dimensions,
            where the first dimension is the batch dimension of size N.

        num_samples : int, default=1
            Number of samples per individual. 

        generator : torch.Generator, optional
            Pseudorandom number generator used for the sampling. If None,
            the default generator is used.
        """

        with torch.no_grad():
            X, T_dummy, _ = self.data_transform(
                X, torch.empty(0), torch.empty(0))

            if len(X.shape) < 2:
                raise ValueError(
                    "The number of dimensions of X must be larger than 1.")

            N = X.shape[0]

            if self.batch_size is None:
                batch_size = N
            else:
                batch_size = self.batch_size

            T = np.empty((N, num_samples))

            for inds in torch.arange(0, N, 1,
                                     dtype=INDEX_DTYPE,
                                     device=T_dummy.device).split(batch_size):

                T[inds, :] = self.torch_model.sample(
                    X[inds], num_samples, generator,**kwargs)

            return T / self.numpy_scale + self.numpy_min_time


class FrozenNumpySurvivalModel():
    r"""Freezes a NumPy model around a set of covariates/features

    Parameters
    ----------

    numpy_model : NumpyModel
        The NumPy model to be frozen
    forward_out : (M,...) Tensor
        The output from the forward function for the covariates/features
        to freeze the model around.
    """

    def __init__(self,
                 numpy_model: NumpySurvivalModel,
                 forward_out: Tensor):
        self.numpy_model = numpy_model
        self.forward_out = forward_out

    def __len__(self):
        return self.forward_out.shape[0]

    def __getitem__(self, inds) -> FrozenNumpySurvivalModel:
        return FrozenNumpySurvivalModel(self.numpy_model,
                                        self.forward_out[inds].reshape(-1, *self.forward_out.shape[1:]))

    def max_time(self) -> float:
        r"""Returns the maximal time after which extrapolation is used."""
        return self.numpy_model.max_time()

    def to_torch(self) -> TorchSurvivalModel:
        r"""Unfreezes the model and returns the underlying PyTorch model"""
        return self.numpy_model.to_torch()

    def unfreeze(self,) -> NumpySurvivalModel:
        r"""Unfreezes the model and returns the underlying NumPy model"""
        return self.numpy_model

    def log_likelihood(self, T: Any, E: Any) -> NDArray:
        r"""Returns the survival function at T.

        Parameters
        ----------

        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)

        E : (N, 1) or (N, 1) or (N,) bool array_like
            Array of event types with the same size as T. Entries evaluated
            as True indicates recorded events and entries evaluated as False
            indicate censored times
            """

        with torch.no_grad():
            _, T, E = self.numpy_model.data_transform(
                torch.empty(0), T, E)

            T = (T-self.numpy_model.numpy_min_time) * \
                self.numpy_model.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, _) = validate_inputs(self.numpy_model.batch_size,
                                                        self.numpy_model.torch_max_time,
                                                        self.forward_out,
                                                        T)

            T = T.reshape(-1,)
            E = E.reshape(-1,)

            return self.numpy_model._log_likelihood(self.forward_out, T, E,
                                                    X_ind_batches, X_out_ind_batches,
                                                    T_ind_batches, T_out_ind_batches,
                                                    output_size, squeeze)

    def density_function(self, T) -> NDArray:
        r"""Returns the density function at T.

        Parameters
        ----------

        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """

        with torch.no_grad():
            _, T, _ = self.numpy_model.data_transform(
                torch.empty(0), T, torch.empty(0))

            T = (T-self.numpy_model.numpy_min_time) * \
                self.numpy_model.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, _) = validate_inputs(self.numpy_model.batch_size,
                                                        self.numpy_model.torch_max_time,
                                                        self.forward_out,
                                                        T)

            T = T.reshape(-1,)

            return self.numpy_model._density_function(self.forward_out, T,
                                                      X_ind_batches, X_out_ind_batches,
                                                      T_ind_batches, T_out_ind_batches,
                                                      output_size, squeeze)

    def survival_function(self, T) -> NDArray:
        r"""Returns the survival function at T.

        Parameters
        ----------

        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """

        with torch.no_grad():
            _, T, _ = self.numpy_model.data_transform(
                torch.empty(0), T, torch.empty(0))

            T = (T-self.numpy_model.numpy_min_time) * \
                self.numpy_model.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, _) = validate_inputs(self.numpy_model.batch_size,
                                                        self.numpy_model.torch_max_time,
                                                        self.forward_out,
                                                        T)

            T = T.reshape(-1,)

            return self.numpy_model._survival_function(self.forward_out, T,
                                                       X_ind_batches, X_out_ind_batches,
                                                       T_ind_batches, T_out_ind_batches,
                                                       output_size, squeeze)

    def hazard_function(self, T: Any) -> NDArray:
        r"""Returns the hazard function at T.

        Parameters
        ----------

        T : (N, 1) or (N, 1) or (N,) array_like
            Array of times to be evaluated. The shape of T should be
            (N,1) (same first dimension of X), (1, M), or (M,), evaluated
            as follows:

            - (N, 1) Each element in the first dimension of X is evaluated with
            the corresponding element in T, creating an output with shape (M,1)
            - (1, M) -- Each element in the first dimension of X is evaluated
            with all elements in T creating an output with shape (N, M)
            - (M,) -- Same as for (1, M), with the extension that if M=1 the
            shape of the output is (M,)
            """

        with torch.no_grad():
            _, T, _ = self.numpy_model.data_transform(
                torch.empty(0), T, torch.empty(0))

            T = (T-self.numpy_model.numpy_min_time) * \
                self.numpy_model.numpy_scale

            ((X_ind_batches, X_out_ind_batches,
              T_ind_batches, T_out_ind_batches),
             output_size, squeeze, _) = validate_inputs(self.numpy_model.batch_size,
                                                        self.numpy_model.torch_max_time,
                                                        self.forward_out,
                                                        T)

            T = T.reshape(-1,)

            return self.numpy_model._hazard_function(self.forward_out, T,
                                                     X_ind_batches, X_out_ind_batches,
                                                     T_ind_batches, T_out_ind_batches,
                                                     output_size, squeeze)

    def sample(self, num_samples: int = 1,
               generator: Optional[torch.Generator] = None,
               **kwargs):
        r"""Generate samples from the distribution

        Parameters
        ----------

        num_samples : int, default=1
            Number of samples per individual. 

        generator : torch.Generator, optional
            Pseudorandom number generator used for the sampling. If None,
            the default generator is used.
        """

        with torch.no_grad():

            N = self.forward_out.shape[0]

            if self.numpy_model.batch_size is None:
                batch_size = N
            else:
                batch_size = self.numpy_model.batch_size

            T = np.empty((N, num_samples))

            for inds in torch.arange(0, N, 1,
                                     dtype=INDEX_DTYPE,
                                     device=self.forward_out.device).split(batch_size):

                T[inds, :] = self.numpy_model.torch_model._sample(self.forward_out[inds],
                                                                  num_samples,
                                                                  generator,**kwargs)

            return T / self.numpy_model.numpy_scale + self.numpy_model.numpy_min_time
