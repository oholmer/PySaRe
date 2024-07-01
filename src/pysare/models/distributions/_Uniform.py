import torch
from pysare.models.SurvivalModel import TorchSurvivalModel


class Uniform(TorchSurvivalModel):
    r"""Uniform distribution, parameters=[low, high]

    Implements a uniform distribution with parameters [low, high] (output from 
    forward function) where low and high are the lower respective upper bound 
    of the support of the distribution.

    """

    def __init__(self) -> None:
        super().__init__()

    def max_time(self):
        return torch.inf

    def _inv_survival_function(self, forward_out, S):
        return forward_out[:, 0]+(1.0-S)*(forward_out[:, 1]-forward_out[:, 0])

    def inv_survival_function(self, X, S):
        T = torch.full_like(S, torch.nan)
        inds = (S >= 0.0) & (S <= 1.0)
        if inds.any():
            T[inds] = self._inv_survival_function(
                self.forward(X[inds]), S[inds])
        return T

    def _survival_function(self, forward_out, t):
        return (1.0-(t-forward_out[:, 0])/(forward_out[:, 1] -
                                           forward_out[:, 0])).clip(0.0, 1.0)

    def _density_function(self, forward_out, t):
        return ((t >= forward_out[:, 0]) & (t <= forward_out[:, 1]))/(forward_out[:, 1] - forward_out[:, 0])

    def _sample(self, forward_out, num_samples=1, generator=None):
        u = torch.rand((forward_out.shape[0], num_samples),
                       device=forward_out.device,
                       dtype=forward_out.dtype,
                       generator=generator)
        return forward_out[:, [0]] + u*(forward_out[:, [1]]-forward_out[:, [0]])

    def forward(self, X):
        return X
