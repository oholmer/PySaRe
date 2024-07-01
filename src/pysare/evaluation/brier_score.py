import time
import torch
import pysare
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import pysare
import numpy as np
import matplotlib.pyplot as plt
import lifelines as lf
import pandas as pd
from collections.abc import Iterable
from matplotlib.axes._axes import Axes
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models._numpy.NumpySurvivalModel import NumpySurvivalModel
from pysare.models._numpy.NumpySurvivalModel import FrozenNumpySurvivalModel
from pysare.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Optional, Tuple, Literal
from numpy.typing import ArrayLike
import torch.utils.data
import pysare
import numpy as np
import matplotlib.pyplot as plt
import lifelines as lf
import pandas as pd
from pysare._typing import X, T, E


def brier_score(model: Union[TorchSurvivalModel, NumpySurvivalModel],
                data: Union[Tuple[X, T, E], Dataset, DataLoader],
                times: Union[T, int],
                integrated: bool = False,
                censoring: Union[Literal['KM', 'none'],
                                 FrozenNumpySurvivalModel] = 'KM',
                plot: Union[bool, Axes] = False) -> pd.DataFrame:
    r""" Brier and integrated Brier score

    Calculates the brier score and integrated brier score. Also considers 
    censoring, see [1] for more information.

    Parameters
    ----------

    model : SurvivalModel or NumpySurvivalModel
        Model to be evaluated. If data is a tuple (X, T, E) of non-tensors, 
        the model should be a NumpySurvivalModel, and otherwise it should 
        be a SurvivalModel.
    data : tuple(X,T,E) or Dataset or Dataloader
        The data to be used for evaluation. If data is a tuple (X, T, E) of 
        non-tensors, the model should be a NumpySurvivalModel, and otherwise 
        it should be a SurvivalModel.
    times : int or (n,) array_like
        Times where the score is evaluated. Should have same data type as 
        the times (T) in the data parameter. If int, this number of linearly
        spaced times between zero and largest time in dataset is used. Times
        where the survival function of the censoring distribution is zero 
        are removed to avoid NaN in the output.
    integrated : bool, default=False
        Indicates if the integrated score should be calculated, False by 
        default.
    censoring : {'KM',  'none'} or FrozenNumpySurvivalModel, default='KM'
        Determines how the censoring distribution is calculated. If 'KM' 
        it is estimated using Kaplan-Meier estimator using the data, if 
        'None' no censoring is use, and if it is a FrozenNumpySurvivalModel
        this model is used.
    plot : bool or Axis, default=False
        If True, a plot of the result will be created after the calculation,
        and if an Axis is passed the plot wil be done in this axis. 
        If False (default), no plot is made.

    Returns
    ----------
        result : DataFrame
            A pandas DataFrame with the result.

    References
    ----------
    [1] Graf, Erika, et al. "Assessment and comparison of prognostic 
    classification schemes for survival data." Statistics in medicine 
    18.17‐18 (1999): 2529-2545.

    """

    if hasattr(model, 'eval'):
        model.eval()

    # Interpret inputs
    # ==========================================================================
    if isinstance(data, tuple):  # data is a tuple (X, T, E)
        dataloader = (data,)
    elif isinstance(data, Iterable):  # data is a DataLoader
        dataloader = data
    else:  # Data is a Dataset
        dataloader = (data[:],)

    dataset = data

    # Full vector of times  and events
    # ===========================================================================
    T = []
    E = []

    for X_batch, T_batch, E_batch in dataloader:
        T.append(T_batch.reshape(-1,))
        E.append(E_batch.reshape(-1,))

    if torch.is_tensor(T[0]):
        T = torch.concat(T)
        E = torch.concat(E)
        T_np = np.array(T)
        E_np = np.array(E)
    else:
        T = np.concatenate(T)
        E = np.concatenate(E)
        T_np = T
        E_np = E

    if isinstance(model, pysare.models.TorchSurvivalModel):
        def get_time_vector(time, length):
            return time.repeat(length)
    else:
        def get_time_vector(time, length):
            return time.repeat(length).reshape(-1, 1)

    if isinstance(times, int):
        if torch.is_tensor(T):
            times = torch.linspace(
                0, max(T), times, dtype=T.dtype, device=T.device)
            times_np = np.array(times.cpu())
        else:
            times = np.linspace(0, max(T), times)
            times_np = times

    if censoring == "KM":
        KMC = lf.KaplanMeierFitter()
        KMC.fit(T_np, ~E_np)
        S_cens_t = KMC.survival_function_at_times(times_np).values
        S_cens_T_inv = 1 / \
            KMC.survival_function_at_times(
                T_np.reshape(-1,)).values.reshape(-1,)
    elif (censoring is False) or (censoring == 'none'):
        S_cens_t = np.ones_like(times_np)
        S_cens_T_inv = np.ones((T.shape[0],))
    else:
        S_cens_t = censoring.survival_function(times)
        S_cens_T_inv = 1/censoring.survival_function(T)

    inds = S_cens_t > 0.0
    times = times[inds]
    times_np = times_np[inds]
    S_cens_t = S_cens_t[inds]

    B = np.zeros_like(times_np)

    with torch.no_grad():
        for n in range(times_np.shape[0]):

            R = []
            for X_batch, _, _ in dataloader:
                R.append(np.array(model.survival_function(
                    X_batch, get_time_vector(times[n:n+1], X_batch.shape[0])).reshape(-1,), copy=False))
            R = np.concatenate(R)

            B1 = ((R**2)*(T_np <= times_np[n])*S_cens_T_inv)[E]

            B[n] = (B1.sum() + ((T_np > times_np[n])*(1 - R)
                    ** 2).sum()/S_cens_t[n])/T_np.shape[0]

    if integrated:
        IB = np.empty_like(B)
        IB[0] = 0.0
        IB[1:] = (B[1:]+B[:-1])*np.diff(times_np)/2
        IB[1:] = IB.cumsum()[1:]/(times_np[1:]-times_np.min())

    # Plot if ax is not False
    if plot:
        color_1, color_2 = plt.rcParams['axes.prop_cycle'].by_key()[
            'color'][:2]

        if plot is True:
            if censoring is not False:
                fig, ax = plt.subplots(nrows=2)
            else:
                fig, ax = plt.subplots(nrows=1)
        else:
            ax = plot

        if censoring is not False:
            if censoring == 'KM':
                KMC.plot_survival_function(ax=ax[1])
            else:
                ax[1].plot(t, S_cens_t)
            ax[1].legend(['Kaplan-Meier Estimate',
                         'Confidence Interval (95%)'])
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Censoring Distribution')
        else:
            ax = [ax]

        ax[0].plot(times_np, B, color=color_1)
        if integrated:
            ax0_twin = ax[0].twinx()
            ax0_twin.plot(times_np, IB, color=color_2)
            ax0_twin.set_ylabel('Integrated Brier Score')

            ax[0].yaxis.label.set_color(color_1)
            ax0_twin.yaxis.label.set_color(color_2)
            ax[0].tick_params(axis='y', colors=color_1)
            ax0_twin.tick_params(axis='y', colors=color_2)

        ax[0].set_ylabel('Brier Score')
        ax[0].set_xlabel('Time')
        plt.tight_layout()

    if integrated:
        return pd.DataFrame(data={'brier_score': B, 'integrated_brier_score': IB,
                                  'time': times_np, 'censoring_survival': S_cens_t}).set_index('time')
    else:
        return pd.DataFrame(data={'brier_score': B, 'time': times_np, 'censoring_survival': S_cens_t}).set_index('time')
