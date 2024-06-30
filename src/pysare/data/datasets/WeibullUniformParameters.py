import numpy as np
from pysare.data import Dataset
import matplotlib.pyplot as plt


class WeibullUniformParameters(Dataset):
    def __init__(self, N=300, t_m=3, scale_range=[1, 3], shape_range=[0.5, 5]):
        

        X, T, E = simulate_weibull_uniform_parameters(N, t_m, scale_range, shape_range)
        
        super(WeibullUniformParameters, self).__init__(X, T, E)
        self.num_features = 2

        self.scale_range = scale_range
        self.shape_range = shape_range
        self.t_m = t_m

    def compare_survival_function(self, models, num_scale, num_shape, num_times=100, ax=None):
        models = np.array(models).reshape(-1,)
        t = np.linspace(0, self.t_m, num_times)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(num_scale, num_shape, clear=True)

        # Make sure ax is two dimensional array
        ax = np.array(ax)
        ax = ax.reshape(num_scale, num_shape)

        for n, la in enumerate(np.linspace(self.scale_range[0],
                                           self.scale_range[1],
                                           num_scale)):
            for m, k in enumerate(np.linspace(self.shape_range[0],
                                              self.shape_range[1],
                                              num_shape)):

                ax[n, m].plot(t, np.exp(-(t/la)**k))
                for model in models:
                    ax[n, m].plot(t, model.survival_probability([[k, la]], t))
                ax[n, m].set_xlim([0, self.t_m])
                ax[n, m].set_ylim([0, 1])
                ax[n,m].set_xlabel('t')
                ax[n,m].set_ylabel('S(t)')

        ax[0,-1].legend(['True','Model'])
        plt.tight_layout()
        return fig, ax

    def compare_model(self, model, scale_parameter, shape_parameter, num_times=100, ax=None):

        t = np.linspace(0, self.t_m, num_times)

        if not ax:
            fig, ax = plt.subplots(2, 1, clear=True)

        ax[0].plot(t, np.exp(-(t/scale_parameter)**shape_parameter))
        ax[0].plot(t, model.survival_probability(
            [[shape_parameter, scale_parameter]], t))

        ax[1].plot(t, shape_parameter/scale_parameter
                   * (t/scale_parameter)**(shape_parameter-1)
                   * np.exp(-(t/scale_parameter)**shape_parameter))
        ax[1].plot(t, model.lifetime_density(
            [[shape_parameter, scale_parameter]], t))

def simulate_weibull_uniform_parameters(N, t_m=3, scale_range=[1, 3], shape_range=[0.5, 5]):
    X1 = np.random.uniform(
        low=shape_range[0], high=shape_range[1], size=[N, 1])
    X2 = np.random.uniform(
        low=scale_range[0], high=scale_range[1], size=[N, 1])
    X = np.concatenate((X1, X2), axis=1)

    def failure_dist(x): return np.random.weibull(x[0]) * x[1]
    def censor_dist(x): return np.random.uniform(low=0, high=t_m)

    T, E = simulate(failure_dist, censor_dist, X)

    return X, T, E


def simulate(FailureDist, CensorDist, V):
    T = np.zeros((V.shape[0], 1))
    C = np.zeros_like(T)

    for n in range(V.shape[0]):
        T[n] = FailureDist(V[n])
        C[n] = CensorDist(V[n])

    E = T <= C
    T = np.concatenate((T, C), axis=1).min(axis=1).reshape(-1,)

    return T, E.reshape(-1,)
