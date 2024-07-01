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
from pysare.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Optional, Tuple, Literal, NewType
from numpy.typing import ArrayLike
from pandas import DataFrame

from pysare._typing import X, T, E


def negative_log_likelihood(
    model: Union[TorchSurvivalModel, NumpySurvivalModel],
        data: Union[Tuple[X, T, E], Dataset, DataLoader],
        plot: Union[bool, Axes, Iterable[Axes]] = False
) -> DataFrame:
    r"""Negative log-likelihood of a model on a dataset

    Calculates the truncated negative log-likelihood for a model on a dataset. 
    Truncated refers to that the mean log-likelihood of all event times up to 
    a specific time is calculated, the last value thus corresponds to the 
    non-truncated value.

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

        plot : bool or Axis, default=False
            If True, a plot of the result will be created in a new figure; if 
            an Axes is passed, the plots will be made in this.

    Returns
    ----------

        result : DataFrame
            A pandas DataFrame with the result.
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

    # Full vector of times and log-likelihoods
    # ===========================================================================
    T = []
    E = []
    l = []
    with torch.no_grad():
        if isinstance(model, pysare.models.TorchSurvivalModel):
            for X_batch, T_batch, E_batch in dataloader:
                T.append(np.array(T_batch.reshape(-1,), copy=True))
                l.append(np.array(model.log_likelihood(
                    X_batch, T_batch, E_batch), copy=True))
        else:
            for X_batch, T_batch, E_batch in dataloader:
                    T.append(np.array(T_batch.reshape(-1,), copy=True))
                    l.append(np.array(model.log_likelihood(
                        X_batch, T_batch.reshape(-1,1), E_batch.reshape(-1,1)), 
                        copy=True))

    T = np.concatenate(T)
    l = np.concatenate(l)

    order = T.argsort()
    T = T[order]
    l = l[order]

    nll = -l.cumsum()/np.arange(1, len(l)+1)


    # Plot if ax is not False
    if plot:

        if plot is True:
            fig, ax = plt.subplots(nrows=1)
        else:
            ax = plot

        # ax.annotate(f'End: {nll[-1]:6.6g}', (T[-1], nll[-1]),
        #      textcoords="offset points", xytext=(-100,0),
        #      ha='center',va='center', arrowprops=dict(arrowstyle="->"))

        ax.plot(T, nll)
        ax.set_ylabel('Truncated negative log-likelihood')
        ax.set_xlabel('Truncation Time')
        
        if plot is True:
            plt.tight_layout()

    return pd.DataFrame(data={'negative_log_lik': nll, 'time': T}).set_index('time')
