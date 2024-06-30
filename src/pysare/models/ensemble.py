from typing import Iterable
from math import floor
import pysare
import numpy as np
import torch
import matplotlib.pyplot as plt
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models._numpy.NumpySurvivalModel import NumpySurvivalModel, FrozenNumpySurvivalModel

def ensemble(models):
    for model in models:
        if isinstance(model, FrozenNumpySurvivalModel):
            return FrozenNumpyEnsembleModel(models)
        elif isinstance(model, NumpySurvivalModel):
            return NumpyEnsembleModel(models)
        else:
            return TorchEnsembleModel(models)


class TorchEnsembleModel(TorchSurvivalModel):
    def __init__(self, models):
        super().__init__()

        self.models = torch.nn.ModuleList()
        for model in models:
            if isinstance(model, TorchSurvivalModel):
                self.models.append(model)
            else:
                raise TypeError('One of the entries in `models`'
                                + ' is no a TorchSurvivalModel')

    def _all_survival_functions(self, X, T):
        return torch.stack([model.survival_function(X, T) for
                            model in self.models])

    def _all_density_functions(self, X, T):
        return torch.stack([model.density_function(X, T) for
                            model in self.models])

    def survival_function(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self._all_survival_functions(X, T).mean(dim=0)

    def density_function(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self._all_density_functions(X, T).mean(dim=0)

    def survival_confidence(self, X, T, confidence_level=0.95, return_survival=False):

        return self._calculate_confidence(
            self._all_survival_functions(X, T).sort(dim=0)[0],
            confidence_level,
            return_survival)

    def density_confidence(self, X, T, confidence_level=0.95, return_survival=False):
        return self._calculate_confidence(
            self._all_density_functions(X, T).sort(dim=0)[0],
            confidence_level,
            return_survival)

    def _calculate_confidence(self, quantity, confidence_level, return_mean):

        alpha_0 = .5*(1.0 - (len(self.models)-2)/(max(1, len(self.models)-1)))

        if isinstance(confidence_level, Iterable):
            alpha = .5 * \
                (1.0-torch.tensor(confidence_level, dtype=torch.float64))
            alpha_index = (alpha/alpha_0).to(dtype=torch.int64)
        else:
            alpha = .5*(1.0-confidence_level)
            alpha_index = int(alpha/alpha_0)

        if return_mean:
            return quantity[alpha_index, :], quantity[-1-alpha_index, :], quantity.mean(dim=0)
        else:
            return quantity[alpha_index, :], quantity[-1-alpha_index, :]

    def forward(self, X):
        return torch.stack([model(X) for model in self.models])


class NumpyEnsembleModel():
    def __init__(self, models) -> None:
        super().__init__()

        self.models = []
        for model in models:
            if isinstance(model, NumpySurvivalModel):
                self.models.append(model)
            else:
                raise TypeError('One of the entries in `models`'
                                + ' is no a NumpySurvivalModel')

    def _all_survival_functions(self, X, T):
        return np.stack(
            [model.survival_function(X,T).reshape(len(X),-1)
             for model in self.models])
    def _all_density_functions(self, X, T):
        return np.stack(
            [model.density_function(X,T).reshape(len(X),-1)
             for model in self.models])

    def survival_function(self,X,T):
        return self._all_survival_functions(X,T).mean(axis=0).squeeze()

    def density_function(self,X,T):
        return self._all_density_functions(X,T).mean(axis=0).squeeze()

    def survival_confidence(self, X, T, confidence_level=0.95, 
                            return_survival=False):
        S = self._all_survival_functions(X, T)
        S.sort(axis=0)
        return self._calculate_confidence(
            S,
            confidence_level,
            return_survival)

    def density_confidence(self, X, T, confidence_level=0.95, 
                            return_density=False):
        S = self._all_density_functions(X, T)
        S.sort(axis=0)
        return self._calculate_confidence(
            S,
            confidence_level,
            return_density)


    def _calculate_confidence(self, quantity, confidence_level, return_mean):

        alpha_0 = .5*(1.0 - (len(self.models)-2)/(max(1, len(self.models)-1)))

        if isinstance(confidence_level, Iterable):
            alpha = .5 * \
                (1.0-np.array(confidence_level))
            alpha_index = (alpha/alpha_0).astype(int)
        else:
            alpha = .5*(1.0-confidence_level)
            alpha_index = int(alpha/alpha_0)

        if return_mean:
            return quantity[alpha_index, :,:], quantity[-1-alpha_index,:,:], quantity.mean(axis=0)
        else:
            return quantity[alpha_index, :,:], quantity[-1-alpha_index, :,:]

class FrozenNumpyEnsembleModel():
    def __init__(self, models) -> None:
        super().__init__()

        self.models = []
        for model in models:
            if isinstance(model, FrozenNumpySurvivalModel):
                self.models.append(model)
            else:
                raise TypeError('One of the entries in `models`'
                                + ' is no a FrozenNumpySurvivalModel')

        self._shape = models[0].forward_out.shape[0]
        for model in models[1:]:
            if model.forward_out.shape[0]!=self._shape:
                raise ValueError('All models must be frozen around the same'+
                                 ' number of features.')            

    def _all_survival_functions(self, T):
        return np.stack(
            [model.survival_function(T).reshape(self._shape,-1)
             for model in self.models])

    def survival_function(self,T):
        return self._all_survival_functions(T).mean(axis=0).squeeze()

    def survival_confidence(self, T, confidence_level=0.95, 
                            return_survival=False):
        S = self._all_survival_functions(T)
        S.sort(axis=0)
        return self._calculate_confidence(
            S,
            confidence_level,
            return_survival)

    def _calculate_confidence(self, quantity, confidence_level, return_mean):

        alpha_0 = .5*(1.0 - (len(self.models)-2)/(max(1, len(self.models)-1)))

        if isinstance(confidence_level, Iterable):
            alpha = .5 * \
                (1.0-np.array(confidence_level))
            alpha_index = (alpha/alpha_0).astype(int)
        else:
            alpha = .5*(1.0-confidence_level)
            alpha_index = int(alpha/alpha_0)

        if return_mean:
            return quantity[alpha_index, :,:], quantity[-1-alpha_index,:,:], quantity.mean(axis=0)
        else:
            return quantity[alpha_index, :,:], quantity[-1-alpha_index, :,:]
        
