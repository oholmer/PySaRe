import torch
import numpy as np
from abc import ABC, abstractmethod


class _Integrator:

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):
        pass

    @abstractmethod
    def integrate(self, model , X, T, t_m, tail_ratio=1.2):
        pass



class MonteCarlo(_Integrator):
    

    def __init__(self, N):
        super(MonteCarlo, self).__init__()
        self.N = N

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):

        return torch.log(self.integrate(model, X, T, t_m, tail_ratio=1.2))

    def integrate(self, model , X, T, t_m, tail_ratio=1.2):



        X_tail = torch.cat((X,T.reshape(-1,1)*0+tail_ratio*t_m), dim=1)
        F_tail = torch.exp(model.forward(X_tail)).reshape(-1,1)

        T = T.reshape(-1,1)
        if X.dim() == 1:
            X = X.reshape(-1, 1)

        dt = torch.kron(t_m-T, torch.ones((self.N, 1)))
        t = torch.rand((self.N*X.shape[0],1))

        X = torch.cat((torch.kron(X, torch.ones((self.N, 1))),
                       dt.reshape(-1,1)*t.reshape(-1, 1)
                       + torch.kron(T.reshape(-1,1), torch.ones((self.N, 1)))
                       ), dim=1)

        F = torch.exp(model.forward(X)).reshape(-1, self.N)

        Z = F.sum(axis=1).reshape(-1, 1)/self.N*(t_m-T) \
            + F_tail*(tail_ratio-1)*t_m

        return Z


class UniformTrapezoidal(_Integrator):
     

    def __init__(self, N):
        super(UniformTrapezoidal, self).__init__()
        
        self.N = N


    def _segment(self, X, T, t_m, tail_ratio):
        if X.dim() == 1:
            X = X.reshape(-1, 1)

        u = torch.ones((self.N,))
        u = u / u.sum()
        #        t = torch.cat((torch.zeros(1), u.cumsum(dim=0)))

        t = torch.cat((torch.zeros(1), u.cumsum(dim=0), torch.ones(1) * tail_ratio))
        # u = torch.cat((u, torch.ones(1) * (self.tail_ratio - 1.)))
        # FIXME: tail not handled correctly below
        X = torch.cat((torch.kron(X, torch.ones((self.N + 2, 1))),
                       torch.kron(t_m - T.reshape(-1, 1), t.reshape(-1, 1))
                       + torch.kron(T.reshape(- 1, 1), torch.ones_like(t.reshape(-1, 1)))
                       ), dim=1)
        X[self.N + 1::self.N + 2, -1] = tail_ratio*t_m

        return X, u

    def integrate(self, model, X, T, t_m, tail_ratio=1.2):
        X, u = self._segment(X, T, t_m, tail_ratio)

        F = torch.exp(model.forward(X)).reshape(-1, self.N + 2)

        Z = ((F[:, :-2] @ u + F[:, 1:-1] @ u) / 2).reshape(-1, 1) \
            * (t_m - T.reshape(- 1
                                    , 1)).reshape(-1, 1) \
            + F[:, -1].reshape(-1, 1) * t_m * (tail_ratio - 1)

        return Z

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):
        X, u = self._segment(X, T, t_m, tail_ratio)

        f = model.forward(X).reshape(-1, self.N + 2)
        f_ast = f.max(dim=1).values
        f -= f_ast[:, None]
        F = torch.exp(f)

        log_Z = f_ast.reshape(-1, 1) \
                + torch.log(((F[:, :-2] @ u
                              + F[:, 1:-1] @ u) / 2).reshape(- 1, 1)
                            * (t_m - T.reshape(-1, 1)).reshape(- 1, 1)
                            + F[:, -1].reshape(-1, 1) * t_m * (tail_ratio - 1))

        return log_Z
