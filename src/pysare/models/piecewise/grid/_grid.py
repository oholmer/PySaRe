from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Any


@dataclass
class Discretization:
    ind: Tensor
    dt: Tensor
    dt_ind: Tensor
    tau_ind: Tensor
    tau_next_ind: Tensor


class DiscretizationGrid(torch.nn.Module):
    r"""Superclass for discretization grids.

    Superclass this class to define a custom grid.    
    """

    @abstractmethod
    def bin(self, t, z):
        pass

    def max_time(self):
        r"""Returns the maximal time in the discretisation."""
        pass

    def num_intervals(self):
        r"""Returns the number of intervals in the grid."""
        pass

    def extra_forward_output_size(self):
        r"""Returns the number elements required by the grid, in adition to 
        those used by the piecewise model."""
        pass


class EquidistantGrid(DiscretizationGrid):
    r"""Implements an uniform grid.

    A fixed grid with equidistant grid points that can not be trained.

    Parameters
    ----------

    max_time : float>0
        The last point in the discretization.
    num_intervals : int>0
        An integer specifying the number of intervals in the discretization.

    """

    def __init__(self, max_time: float, num_intervals: int):
        super().__init__()

        # num_intervals is and int and is handled using bufffer to remain so
        self._num_intervals = torch.nn.Parameter(torch.tensor(
            num_intervals).type(torch.long), requires_grad=False)

        # dt is a float and is handled as a Parameter so that it changes type with model
        self.dt = torch.nn.Parameter(torch.tensor(max_time/num_intervals,
                                                  dtype=torch.get_default_dtype()),
                                     requires_grad=False)

    def extra_forward_output_size(self):
        return 0

    def num_intervals(self):
        return self._num_intervals

    def to(self, *args, **kwargs):
        rval = super().to(*args, **kwargs)

        # N should remain long
        self._num_intervals = self._num_intervals.to(dtype=torch.long)
        return rval

    def max_time(self):
        return (self._num_intervals*self.dt).item()

    def bin(self, t, z):

        ind = torch.clip(torch.floor_divide(t, self.dt).to(
            torch.long), min=None, max=self._num_intervals-1)

        return Discretization(
            ind=ind,
            dt=self.dt,
            dt_ind=self.dt,
            tau_ind=self.dt*ind,
            tau_next_ind=self.dt*(ind+1)), z


class FixedGrid(DiscretizationGrid):
    r"""Implements a fixed grid.

    A fixed grid where the discretization is provided by the user, and can not 
    be trained.

    Parameters
    ----------
    discretization_times : (num_intervals+1,) array-like of increasing floats
        Discretization times. Will be converted to a tensor with default dtype 
        and device. The times must be increasing, no test of this is done.
    """

    def __init__(self, discretization_times: Any):
        super().__init__()

        self.discretization_times = torch.nn.Parameter(torch.tensor(
            discretization_times,
            dtype=torch.get_default_dtype()), requires_grad=False)
        self._dt = torch.nn.Parameter(torch.diff(self.discretization_times),
                                      requires_grad=False)

    def extra_forward_output_size(self):
        return 0

    def to(self, *args, **kwargs):
        rval = super().to(*args, **kwargs)
        return rval

    def max_time(self):
        return self.discretization_times[-1].item()

    def num_intervals(self):
        return self.discretization_times.shape[0]-1

    def bin(self, t, z):

        N = t.shape[0]

        ind = torch.searchsorted(
            self.discretization_times, t,
            right=True).clip(None,
                             self.discretization_times.shape[0]-1)

        tau_next_ind = self.discretization_times[ind]
        ind -= 1
        tau_ind = self.discretization_times[ind]

        # ind = torch.clip(torch.floor_divide(t, self.dt).to(
        # torch.long), min=None, max=self.N-1)

        return Discretization(
            ind=ind,
            dt=self._dt,
            dt_ind=self._dt[ind],
            tau_ind=tau_ind,
            tau_next_ind=tau_next_ind), z


class TrainableGrid(DiscretizationGrid):
    r"""Implements a trainable grid.

    A trainable grid where the discretization is parameterized by a tensor 
    which, after applying softmax and normalizing so that they sum up to the
    maximal time, specifies the size of each interval.

    Parameters
    ----------
    discretization_times : (num_intervals+1,) array-like of increasing floats
        Discretization times. Will be used to determine the maximal time (last
        value) and the initial parameterization of the grid.
    """

    def __init__(self, discretization_times: Any):
        super().__init__()

        self._max_time = float(discretization_times[-1])

        log_prop_dt = torch.log(torch.diff(
            torch.tensor(discretization_times,
                         dtype=torch.get_default_dtype())))

        self.log_prop_dt = torch.nn.Parameter(
            log_prop_dt-torch.max(log_prop_dt),
            requires_grad=True)

    def extra_forward_output_size(self):
        return 0

    def to(self, *args, **kwargs):
        rval = super().to(*args, **kwargs)
        return rval

    def max_time(self):
        return self._max_time

    def num_intervals(self):
        return self.log_prop_dt.shape[0]

    def discretization_times(self):
        delta = torch.softmax(self.log_prop_dt, 0)
        dt = self._max_time*delta
        tau = dt.cumsum(0)

        return torch.concat((torch.zeros((1,), dtype=tau.dtype,
                                         device=tau.device),
                             tau))

    def bin(self, t, z):

        N = t.shape[0]

        delta = torch.softmax(self.log_prop_dt, 0)

        dt = self._max_time*delta

        tau = dt.cumsum(0)

        ind = torch.searchsorted(
            tau, t, right=False).clip(None, tau.shape[0]-1)

        tau_next_ind = tau[ind]
        tau_ind = tau_next_ind - dt[ind]

        return Discretization(
            ind=ind,
            dt=dt,
            dt_ind=dt[ind],
            tau_ind=tau_ind,
            tau_next_ind=tau_next_ind), z


class IndividualizedGrid(DiscretizationGrid):
    r"""Implements a grid where the discretization depends on the features.

    This grid uses the output from the forward function of the model to 
    determine the discretization individualy for each sampel.

    To use this grid, the forward functoin of the model shold return 
    num_intervals extra entries, positioned last in the tensor. After applying 
    softmax, theses are used to determine the size of each interval so that 
    they sum up to max_time. 

    Parameters
    ----------
    max_time : float>0
        The last point in the discretization.
    num_intervals : int>0
        An integer specifying the number of intervals in the discretization.
    """

    def __init__(self, max_time: float, num_intervals: int):
        super().__init__()

        # num_intervals is and int and is handled using bufffer to remain so
        self._num_intervals = torch.nn.Parameter(torch.tensor(
            num_intervals).type(torch.long), requires_grad=False)

        self._max_time = float(max_time)

    def extra_forward_output_size(self):
        return self._num_intervals

    def max_time(self):
        return self._max_time

    def num_intervals(self):
        return self._num_intervals

    def bin(self, t, z):

        # The last self.num_intervals entries in z (dim 1) should be used here
        v = z[:, -self._num_intervals:]
        z = z[:, :-self._num_intervals]

        N = t.shape[0]

        delta = torch.softmax(v, 1)

        dt = self._max_time*delta
        tau = dt.cumsum(1)

        ind = (tau[:, :-1] < t.reshape(-1, 1)).sum(dim=1)

        tau_ind_next = tau[range(N), ind]
        tau_ind = tau_ind_next - dt[range(N), ind]

        # ind = torch.clip(torch.floor_divide(t, self.dt).to(
        # torch.long), min=None, max=self.N-1)

        return Discretization(
            ind=ind,
            dt=dt,
            dt_ind=dt[range(N), ind],
            tau_ind=tau_ind,
            tau_next_ind=tau_ind_next), z
