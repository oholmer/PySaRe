from torch.nn import Module
import torch


class SoftmaxMixer(Module):
    def __init__(self) -> None:
        super().__init__()

    def _calc_weights(self, log_prop_weights):
        weights = torch.exp(log_prop_weights)
        return weights/weights.sum(dim=1).reshape(-1, 1)

    def _calc_prop_weights(self, log_prop_weights):
        prop_weights = torch.exp(
            log_prop_weights - torch.max(log_prop_weights, dim=1, keepdim=True)[0])
        return prop_weights

    def survival_probability(self, T, distributions, parameters, log_prop_weights):
        weights = self._calc_weights(log_prop_weights)

        S = distributions.survival_probability(parameters, T)
        return (S*weights).sum(dim=1)

    def log_survival_probability(self, T, distributions, parameters, log_prop_weights):
        prop_weights = self._calc_prop_weights(log_prop_weights)

        S = distributions.survival_probability(parameters, T)
        return torch.log((S*prop_weights).sum(dim=1)) - torch.log(prop_weights.sum(dim=1))

    def lifetime_density(self, T, distributions, parameters, log_prop_weights):
        weights = self._calc_weights(log_prop_weights)

        f = distributions.lifetime_density(parameters, T)
        return (f*weights).sum(dim=1)

    def log_lifetime_density(self, T, distributions, parameters, log_prop_weights):
        prop_weights = self._calc_prop_weights(log_prop_weights)

        f = distributions.lifetime_density(parameters, T)
        return torch.log((f*prop_weights).sum(dim=1)) - torch.log(prop_weights.sum(dim=1))


class SoftmaxMixerWithReflection(SoftmaxMixer):
    def __init__(self) -> None:
        super().__init__()

    def survival_probability(self, T, distributions, parameters, log_prop_weights):
        weights = self._calc_weights(log_prop_weights)

        S = distributions.survival_probability(
            parameters, T) + (1-distributions.survival_probability(parameters, -T))

        return (S*weights).sum(dim=1)

    def log_survival_probability(self, T, distributions, parameters, log_prop_weights):
        prop_weights = self._calc_prop_weights(log_prop_weights)

        S = distributions.survival_probability(
            parameters, T) + (1-distributions.survival_probability(parameters, -T))

        return torch.log((S*prop_weights).sum(dim=1)) - torch.log(prop_weights.sum(dim=1))

    def lifetime_density(self, T, distributions, parameters, log_prop_weights):
        weights = self._calc_weights(log_prop_weights)

        f = distributions.lifetime_density(
            parameters, T)+distributions.lifetime_density(parameters, -T)
        return (f*weights).sum(dim=1)

    def log_lifetime_density(self, T, distributions, parameters, log_prop_weights):
        prop_weights = self._calc_prop_weights(log_prop_weights)

        f = distributions.lifetime_density(
            parameters, T)+distributions.lifetime_density(parameters, -T)
        return torch.log((f*prop_weights).sum(dim=1)) - torch.log(prop_weights.sum(dim=1))
