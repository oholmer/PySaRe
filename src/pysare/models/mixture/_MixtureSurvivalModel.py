import torch
from pysare.models.SurvivalModel import TorchSurvivalModel

class MixtureSurvivalModel(TorchSurvivalModel):
    def __init__(self, distributions, mixer) -> None:
        super().__init__()
        self.mixer = mixer
        self.distributions = distributions

    def max_time(self):
        return torch.inf

    def _survival_function(self, forward_out, T):
        ind = forward_out.shape[1]//(self.distributions.num_parameters+1)*self.distributions.num_parameters
        return self.mixer.survival_probability(T, self.distributions, forward_out[:, :ind], forward_out[:, ind:])
    
    def _log_survival_function(self, forward_out, T):
        ind = forward_out.shape[1]//(self.distributions.num_parameters+1)*self.distributions.num_parameters
        return self.mixer.log_survival_probability(T, self.distributions, forward_out[:, :ind], forward_out[:, ind:])

    def _density_function(self, forward_out, T):
        ind = forward_out.shape[1]//(self.distributions.num_parameters+1)*self.distributions.num_parameters
        return self.mixer.lifetime_density(T, self.distributions, forward_out[:, :ind], forward_out[:, ind:])

    def _log_density_function(self, forward_out, T):
        ind = forward_out.shape[1]//(self.distributions.num_parameters+1)*self.distributions.num_parameters
        return self.mixer.log_lifetime_density(T, self.distributions, forward_out[:, :ind], forward_out[:, ind:])
    
    def forward(self, X):
        return X

