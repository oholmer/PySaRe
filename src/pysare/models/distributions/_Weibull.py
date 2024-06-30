import torch
from pysare.models.SurvivalModel import TorchSurvivalModel
from pysare.models._numpy.NumpySurvivalModel import NumpySurvivalModel

class Weibull(TorchSurvivalModel):
    def __init__(self) -> None:
        super().__init__()

    def _sample(self, forward_out, num_samples, generator=None):
        return torch.distributions.Weibull(forward_out[:,1], forward_out[:,0]).sample(num_samples)
    
    def max_time(self):
        return torch.inf
    
    def _inv_survival_function(self, forward_out, S):
        return forward_out[:, 1]*(-torch.log(S))**(1/forward_out[:, 0])

    def inv_survival_function(self, X, S):
        return self._inv_survival_function(self.forward(X), S)

    def _survival_function(self, forward_out, t):
        return torch.exp(-((t / forward_out[:,1]) ** forward_out[:,0]))

    def _density_function(self, forward_out, t):
        return (
            (forward_out[:,0] / forward_out[:,1])
            * (t / forward_out[:,1]) ** (forward_out[:,0] - 1)
            * torch.exp(-((t / forward_out[:,1]) ** forward_out[:,0]))
        )
    
    def _sample(self, forward_out, num_samples=1, generator=None):
        u = torch.rand((forward_out.shape[0]*num_samples,),
                       device=forward_out.device,
                       dtype=forward_out.dtype,
                       generator=generator)
        return self._inv_survival_function(
            forward_out.repeat_interleave(num_samples, 0),
            u).reshape(-1, num_samples)

    def forward(self, X):
        return X
