import numpy as np
import torch
from torch import nn
import torch
from pysare.models._SurvivalModel import _SurvivalModel

class EBM(_SurvivalModel):

    def __init__(self, t_m, tail_ratio, train_integrator, eval_integrator):
        super(EBM, self).__init__()

        self.t_m = t_m
        self.tail_ratio = tail_ratio

        self._eval_integrator = eval_integrator
        self._train_integrator = train_integrator
        self._active_integrator = train_integrator

    def train(self, mode: bool = True):
        super(EBM, self).train(mode)
        if mode:
            self._active_integrator = self._train_integrator
        else:
            self._active_integrator = self._eval_integrator
        return self

    def eval(self):
        return self.train(False)

    def _log_likelihood(self, X, T, E):

        log_Z0 = self._active_integrator.log_integrate(
            self, X, torch.zeros((T.shape[0], 1)), self.t_m, self.tail_ratio)

        log_Z = self._active_integrator.log_integrate(self, X, T, self.t_m, self.tail_ratio)

        z = self.forward(torch.cat((X, T.reshape(-1, 1)), dim=1))



        l = - log_Z0
        l[E] += z[E]
        l[~E] += log_Z[~E]
        return l
        # return z*E.reshape(-1, 1) \
        #     + log_Z*(1-E).reshape(-1, 1)  \
        #     - log_Z0

    def _lifetime_density(self, X, T):
        Z = self._active_integrator.integrate(
            self, X, torch.zeros((T.shape[0], 1)), self.t_m, self.tail_ratio)

        f = torch.exp(self.forward(torch.cat((X, T.reshape(-1, 1)), dim=1))) \
            / Z.reshape(-1, 1)
        return f

    def _survival_function(self, X, T):
        Z0 = self._active_integrator.integrate(
            self, X, torch.zeros((T.shape[0], 1)), self.t_m, self.tail_ratio)
        Z = self._active_integrator.integrate(self, X, T, self.t_m, self.tail_ratio)
        return Z/Z0

    def forward(self, X):
        return X

    @classmethod
    def MLP_implementation(cls, t_m, tail_ratio, train_integrator, eval_integrator,
                           num_features, layers,
                           activation=torch.nn.ReLU, dropout=False,
                           batch_norm=False):

        return MLP_EBM(t_m, tail_ratio, train_integrator, eval_integrator,
                       num_features, layers,
                       activation=torch.nn.ReLU, dropout=0.0,
                       batch_norm=False)


class MLP_EBM(EBM):
    def __init__(self, t_m, tail_ratio, train_integrator, eval_integrator,
                 num_features, layers,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        super(MLP_EBM, self).__init__(t_m, tail_ratio,
                                  train_integrator, eval_integrator)
        self.num_features = num_features

        input_size = num_features+1
        layerlist = []
        for output_size in layers[:-1]:
            layerlist.append(torch.nn.Linear(input_size, output_size))
            layerlist.append(activation())
            if batch_norm:
                layerlist.append(torch.nn.BatchNorm1d(output_size))
            if dropout:
                layerlist.append(torch.nn.Dropout(dropout))

            input_size = output_size

        layerlist.append(torch.nn.Linear(layers[-1], 1))

        self.layers = torch.nn.Sequential(*layerlist)

    def forward(self, X):
        logits = self.layers(X)

        return logits


# class EBM_old(_SurvivalModel):
#     def __init__(self, train_integrator, eval_integrator):
#         super(EnergyBased, self).__init__()
#         self.loss = LossFunctions(self)
#         self.validate = ValidationMethods(self)

#         self._eval_integrator   = eval_integrator
#         self._train_integrator  = train_integrator
#         self._active_integrator = train_integrator

#     def train(self, mode: bool = True):
#         super(EnergyBased, self).train(mode)
#         if mode:
#             self._active_integrator = self._train_integrator
#         else:
#             self._active_integrator = self._eval_integrator
#         return self

#     def eval(self):
#         return self.train(False)

#     def _likelihood(self, X, H):
#         Z = self._active_integrator.integrate(self, X, torch.zeros((H.shape[0], 1)))

#         f = torch.exp(self.forward(torch.cat((X, H[:,0].reshape(-1, 1)), dim=1)))\
#             / Z.reshape(-1, 1)
#         return f

#     @abstractmethod
#     def _log_likelihood(self, X, H):
#         log_Z0 = self._active_integrator.log_integrate(self, X, torch.zeros((H.shape[0], 1)))

#         log_Z = self._active_integrator.log_integrate(self, X, H[:, 0])

#         z = self.forward(torch.cat((X, H[:, 0].reshape(-1, 1)), dim=1))

#         return z*H[:,1].reshape(-1,1) \
#                + log_Z*(1-H[:,1]).reshape(-1,1)  \
#                - log_Z0


#     @abstractmethod
#     def _lifetime_density(self, X, T):
#         Z = self._active_integrator.integrate(self, X, torch.zeros((T.shape[0], 1)))

#         f = torch.exp(self.forward(torch.cat((X, T.reshape(-1, 1)), dim=1))) \
#             / Z.reshape(-1, 1)
#         return f
#     @abstractmethod
#     def _survival_probability(self, X, T):

#         Z0 = self._active_integrator.integrate(self, X, torch.zeros((T.shape[0], 1)))
#         Z  = self._active_integrator.integrate(self, X, T)
#         return Z/Z0

#     def _random_trapetzoidal(self, X, T):

#         if X.dim() == 1:
#             X = X.reshape(-1, 1)

#         u = torch.rand((self.N,))
#         u = u / u.sum()
#         t = torch.cat((torch.zeros(1), u.cumsum(dim=0)))


#         X = torch.cat((torch.kron(X, torch.ones((self.N + 1, 1))),
#                        torch.kron(self.t_m-T.reshape(-1, 1), t.reshape(-1, 1))
#                        +torch.kron(T.reshape(-1,1), torch.ones_like(t.reshape(-1, 1)))
#                        ), dim=1)

#         F = torch.exp(self.forward(X)).reshape(-1, self.N + 1)

#         Z = ((F[:, :-1] @ u + F[:, 1:] @ u) / 2).reshape(-1,1)\
#             *(self.t_m-T.reshape(-1, 1)).reshape(-1,1)

#         return Z

#     def _trapetzoidal(self, X, T):

#         if X.dim() == 1:
#             X = X.reshape(-1, 1)

#         u = torch.ones((self.N,))
#         u = u / u.sum()
#         t = torch.cat((torch.zeros(1), u.cumsum(dim=0)))


#         X = torch.cat((torch.kron(X, torch.ones((self.N + 1, 1))),
#                        torch.kron(self.t_m-T.reshape(-1, 1), t.reshape(-1, 1))
#                        +torch.kron(T.reshape(-1,1), torch.ones_like(t.reshape(-1, 1)))
#                        ), dim=1)

#         F = torch.exp(self.forward(X)).reshape(-1, self.N + 1)

#         Z = ((F[:, :-1] @ u + F[:, 1:] @ u) / 2).reshape(-1,1)\
#             *(self.t_m-T.reshape(-1, 1)).reshape(-1,1)

#         return Z

#     def _log_trapetzoidal(self, X, T):

#         if X.dim() == 1:
#             X = X.reshape(-1, 1)

#         u = torch.ones((self.N,))
#         u = u / u.sum()
#         t = torch.cat((torch.zeros(1), u.cumsum(dim=0)))


#         X = torch.cat((torch.kron(X, torch.ones((self.N + 1, 1))),
#                        torch.kron(self.t_m-T.reshape(-1, 1), t.reshape(-1, 1))
#                        +torch.kron(T.reshape(-1,1), torch.ones_like(t.reshape(-1, 1)))
#                        ), dim=1)


#         f = self.forward(X).reshape(-1, self.N+1)
#         f_ast = f.max(dim = 1).values
#         f -= f_ast[:, None]
#         F = torch.exp(f)

#         log_Z = f_ast.reshape(-1, 1) + torch.log(((F[:, :-1] @ u + F[:, 1:] @ u) / 2).reshape(-1,1)\
#             *(self.t_m-T.reshape(-1, 1)).reshape(-1,1))

#         return log_Z
#     def _monte_carlo(self, X, T):

#         T = T.reshape(-1,1)
#         if X.dim() == 1:
#             X = X.reshape(-1, 1)


#         dt = torch.kron(self.t_m-T, torch.ones((self.N, 1)))
#         t = torch.rand((self.N*X.shape[0],1))


#         X = torch.cat((torch.kron(X, torch.ones((self.N, 1))),
#                        dt.reshape(-1,1)*t.reshape(-1, 1)
#                        +torch.kron(T.reshape(-1,1), torch.ones((self.N,1)))
#                        ), dim=1)

#         F = torch.exp(self.forward(X)).reshape(-1, self.N )

#         Z = F.sum(axis=1).reshape(-1,1)/self.N*(self.t_m-T)

#         return Z

#    def _metropolis_Hastings(self, X, T):

#        dt = self.t_m-T.reshape(-1, 1)

#        X = np.random.uniform(w[0], w[1], (N,))
#        X[0] = x0
#
#        for n in range(1, N):
#
#            a = 0
#            u = 1
#            while True:
#                x = np.random.uniform(w[0], w[1])
#                a = f(x) / f(X[n - 1])
#                u = np.random.uniform(0, 1)
#                if u <= a:
#                    X[n] = x
#                    break
#                X[n] = X[n - 1]
#                break
#
#        T = T.reshape(-1,1)
#        if X.dim() == 1:
#            X = X.reshape(-1, 1)
#
#        dt = self.t_m-T
#        t = torch.rand((self.N,1))
#
#
#
#        X = torch.cat((torch.kron(X, torch.ones((self.N, 1))),
#                       torch.kron(dt, t.reshape(-1, 1))
#                       +torch.kron(T.reshape(-1,1), torch.ones_like(t.reshape(-1, 1)))
#                       ), dim=1)
#
#        F = torch.exp(self.forward(X)).reshape(-1, self.N )
#
#        Z = F.sum(axis=1).reshape(-1,1)/self.N*dt
#
#        return Z
