
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import pysare
import numpy as np
import matplotlib.pyplot as plt
import lifelines as lf
import pandas as pd
from pandas import DataFrame
from collections.abc import Iterable
from matplotlib.axes._axes import Axes
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models._numpy.NumpySurvivalModel import FrozenNumpySurvivalModel
from pysare.models._numpy.NumpySurvivalModel import NumpySurvivalModel
from pysare.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Optional, Tuple, Literal
from numpy.typing import ArrayLike


from pysare._typing import X, T, E


def concordance_index(model: Union[TorchSurvivalModel, NumpySurvivalModel],
                      data: Union[Tuple[X, T, E], Dataset, DataLoader],
                      censoring: Union[Literal['KM', 'none'],
                                       FrozenNumpySurvivalModel] = 'KM',
                      plot: Union[bool, Axes, Iterable[Axes]] = False) -> DataFrame:
    r"""Concordance index of a model on a dataset.

    Calculates the censoring-corrected truncated time-dependent 
    concordance index (C-index), where: 

    - Time-dependent refers to that no PH assumption is made and instead the 
      comparison is done based on the value of the predicted survival function 
      at the time of one of the event times that are compared. 
    - Censoring-corrected refers to that inverse probability weights are used to 
      make the result less dependent on the censoring distribution
    - Truncated refers to that the result is truncated up to a specific time so 
      that only data points with event times before this time are compared (the 
      value at the largest truncation time corresponds to the conventional index). 

    See references [1] and  [2] for more information.

    Parameters
    ----------

    model : SurvivalModel or NumpySurvivalModel:
        Model to be evaluated. If data is a tuple (X, T, E) of non-tensors, 
        the model should be a NumpySurvivalModel, and otherwise it should 
        be a SurvivalModel.
    data : tuple(X,T,E) or Dataset or Dataloader:
        The data to be used for evaluation. If data is a tuple (X, T, E) of 
        non-tensors, the model should be a NumpySurvivalModel, and otherwise 
        it should be a SurvivalModel.
    censoring :  'KM' or 'none' or FrozenNumpySurvivalModel, default='KM'
        Determines how the censoring distribution is calculated. If 'KM' 
        it is estimated using the Kaplan-Meier estimate of the data, if 
        'None' no censoring is use, and if it is a FrozenNumpySurvivalModel
        this model is used.
    plot : bool or Axis or Iterable[Axes], default=False
        If True, a plot of the result (index and censoring) will be created 
        in a new figure; if one or two Axes are passed, the plots will be 
        made in these.

    Returns
    ----------

        result : DataFrame
            A pandas DataFrame with the result.

    References
    ----------

    [1] Antolini, L., Boracchi, P. and Biganzoli, E. (2005), A time-dependent 
    discrimination index for survival data. Statist. 
    https://doi.org/10.1002/sim.2427

    [2] Uno, H., Cai, T., Pencina, M.J., D'Agostino, R.B. and Wei, L.J. (2011), 
    On the C-statistics for evaluating overall adequacy of risk prediction 
    procedures with censored survival data. Statist. Med., 30: 1105-1117. 
    https://doi.org/10.1002/sim.4154

    """

    if hasattr(model, 'eval'):
        model.eval()

    # Interpret inputs
    # ==========================================================================
    if type(data) is tuple:  # data is a tuple (X, T, E)
        dataloader = (data,)
    elif isinstance(data, Iterable):  # data is a DataLoader
        dataloader = data
    else:  # Data is a Dataset
        dataloader = (data[:],)

    dataset = data

    if isinstance(model,pysare.models.TorchSurvivalModel):
        def get_time_vector(time, length):
            return time.repeat(length)
    else:   
        def get_time_vector(time, length):
            return time.repeat(length).reshape(-1,1)

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

    # Event times (not censorings)
    T_event = T_np[E_np]

    # Uniqe times between the first and last recorded event
    tau = np.sort(
        np.unique(T_np[(T_np <= T_event.max()) & (T_np >= T_event.min())]))

    # Censoring
    # =========================================================================
    if censoring == "KM":  # Censoring estimated using Kaplan-Meier
        KMC = lf.KaplanMeierFitter()
        KMC.fit(T_np, ~E_np)
        C_T = KMC.survival_function_at_times(T_np).values.reshape(-1,)
        C_tau = KMC.survival_function_at_times(tau).values.reshape(-1,)
    elif (censoring == 'none') or (censoring is False):  # No censoring / censoring = 1
        C_T = np.ones((T_np.shape[0],))
        C_tau = np.ones((tau.shape[0],))
    else:  # censoring is a FrozenSurvivalModel
        C_T = censoring.survival_function(T).reshape(-1,)
        C_tau = censoring.survival_function(tau).reshape(-1,)

    # =========================================================================

    A = np.zeros_like(T_np)
    B = np.zeros_like(T_np)
    with torch.no_grad():
        for n in range(T.shape[0]):

            S_Tn = []
            for X_batch, _, _ in dataloader:
                # with torch.no_grad():
                S_Tn.append(np.array(model.survival_function(
                    X_batch, get_time_vector(T[n:n+1],X_batch.shape[0])).flatten(), copy=False))

            # S_Tn = [np.array(model.survival_probability(X_batch,
            #                                             T[n:n+1].repeat(X_batch.shape[0])),
            #                  copy=False)
            #         for X_batch, _, _ in dataloader]
            S_Tn = np.concatenate(S_Tn)

            comp = (T_np[n] < T_np)  # .reshape(-1,)

            A[n] = ((S_Tn[n] < S_Tn) & comp).sum()
            B[n] = comp.sum()

            # A[n] = ((S_Tn[n] < S_Tn) & comp).sum()/(C_T[n]**2)*E_np[n]
            # B[n] = comp.sum()/(C_T[n]**2)*E_np[n]

    A /= (C_T**2)
    A[~E_np] = 0.
    B /= (C_T**2)
    B[~E_np] = 0.

    C = np.zeros(tau.shape)

    for n in range(tau.shape[0]):
        ind = T_np <= tau[n]
        C[n] = A[ind].sum()/B[ind].sum()

    # Plot
    if plot is not False:
        if plot is True:
            if censoring is not False:
                fig, ax = plt.subplots(nrows=2)
            else:
                fig, ax = plt.subplots(nrows=1)
                ax = (ax,)

        elif isinstance(plot, Iterable):
            ax = tuple(plot)
        else:
            ax = (plot,)

        if (censoring is not False) and (len(ax) > 1):
            if censoring == 'KM':
                KMC.plot_survival_function(ax=ax[1])
                ax[1].legend(['Kaplan-Meier Estimate',
                              'Confidence Interval (95%)'])
            else:
                ax[1].plot(T, C_T)

            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Censoring Distribution')

        ax[0].plot(tau, C)
        ax[0].set_ylabel('Truncated C-index')
        ax[0].set_xlabel('Truncation time')

        if plot is True:
            plt.tight_layout()

    return pd.DataFrame(data={'C_index': C, 'truncation_time': tau, 'censoring_survival': C_tau})
