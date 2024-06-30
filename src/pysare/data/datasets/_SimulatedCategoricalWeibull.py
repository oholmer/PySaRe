from numpy import linspace
import torch
from pysare.models.distributions import Uniform, Weibull
from pysare.data.Dataset import Dataset


class CategoricalWeibull(Weibull):
    def __init__(self, shape_categories, scale_categories) -> None:
        super().__init__()

        self.shape_categories = torch.nn.Parameter(
            shape_categories, requires_grad=False)
        self.scale_categories = torch.nn.Parameter(
            scale_categories, requires_grad=False)

    def forward(self, X):
        return torch.concat((self.shape_categories[X[:, [0]]],
                             self.scale_categories[X[:, [1]]]), 1)


def seed_to_generator(seed):

    if isinstance(seed, torch.Generator):
        return seed
    elif seed is None:
        return None
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator


class SimulatedCategoricalWeibull():
    def __init__(self,
                 shuffle_categories=False,
                 shape_categories=linspace(1.5, 5.0, 10),
                 scale_categories=linspace(1, 3.0, 10),
                 max_censoring_time=3.0,
                 seed=None,
                 dtype=torch.get_default_dtype()) -> None:

        generator = seed_to_generator(seed)

        self.shape_categories = torch.tensor(shape_categories, dtype=dtype)
        self.scale_categories = torch.tensor(scale_categories, dtype=dtype)

        self.num_categories = {'shape': len(self.shape_categories),
                               'scale': len(self.scale_categories)}
        
        self.max_censoring_time = max_censoring_time

        if shuffle_categories:
            self.shape_categories = self.shape_categories[
                torch.randperm(len(self.shape_categories),
                               generator=generator)]
            self.shape_categories = self.scale_categories[
                torch.randperm(len(self.scale_categories),
                               generator=generator)]

        float_tensor = torch.empty(0, dtype=dtype)
        int_tensor = torch.empty(0, dtype=torch.int64)

        self.event_distribution = CategoricalWeibull(
            shape_categories=self.shape_categories,
            scale_categories=self.scale_categories).to_numpy(
                data=(int_tensor, float_tensor, float_tensor))

        self.censoring_distribution = Uniform().to_numpy(
            data=(float_tensor,
                  float_tensor,
                  float_tensor)).freeze([[0.0, max_censoring_time]])

    def sample_features(self, num_samples, seed=None):

        generator = seed_to_generator(seed)

        return torch.concat((torch.randint(0, len(self.shape_categories),
                                           (num_samples, 1),
                                           generator=generator),
                             torch.randint(0, len(self.scale_categories),
                                           (num_samples, 1),
                                           generator=generator)), 1)

    def generate(self, num_samples, seed=None):
        generator = seed_to_generator(seed)
        X = self.sample_features(num_samples, seed=generator)
        T_event = self.event_distribution.sample(X, num_samples=1,
                                                 generator=generator).squeeze()
        T_cens = self.censoring_distribution.sample(num_samples=num_samples,
                                                    generator=generator).squeeze()
        E = T_event <= T_cens
        T = T_event
        T[~E] = T_cens[~E]
        return Dataset(X, T, E, copy=False)
