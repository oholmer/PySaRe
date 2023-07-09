import numpy as np
import torch
from torch import nn
from abc import abstractmethod
import pandas as pd
from .default_input_function import default_input_function


class _SurvivalModel(nn.Module):
    def __init__(self, input_function=default_input_function):
        super(_SurvivalModel, self).__init__()
        self.input_function = input_function

    def log_likelihood(self, X, T, E, batch_size=None) -> np.array:
        """Returns an N by M array of log lieklihoods given X, T_a, and E_a.
           
        Args:
            X: Arraylike with features/covariate with X.shape[0] = N.
            T: Arraylike with times used to determine T_a
               If T.shape[0]=1, T_a[i,j] = T[1,j].
               If T.shape[0]=N (same as X.shape[0]), T_a[i,j] = T[i,j]
            E: Arraylike with event types used to determine E_a in the same 
               way as T_a is based on T"""

        self.eval()
        X, T, E, output_function = self.input_function(self, X, T, E)
        return output_function(self._log_likelihood(X, T, E))

    def lifetime_density(self, X, T, batch_size=None) -> np.array:
        """Returns an N by M array of lifetime densities given X and T_a.
           
        Args:
            X: Arraylike with features/covariate with X.shape[0] = N.
            T: Arraylike with times used to determine T_a
               If T.shape[0]=1, T_a[i,j] = T[1,j]."""
        
        self.eval()
        X, T, output_function = self.input_function(self, X, T)
        return output_function(self._lifetime_density(X, T).detach())

    def survival_probability(self, X, T, batch_size=None) -> np.array:
        """Returns an N by M array of survival probabilities given X and T_a.
           
        Args:
            X: Arraylike with features/covariate with X.shape[0] = N.
            T: Arraylike with times used to determine T_a
               If T.shape[0]=1, T_a[i,j] = T[1,j]."""
        
        self.eval()
        X, T, output_function = self.input_function(self, X, T)
        return output_function(self._survival_function(X, T))
    
