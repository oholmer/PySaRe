import torch
import numpy as np
from abc import ABC, abstractmethod


class EnergyIntegrator:
    r"""Base class for energy integrators."""

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):
        pass

    @abstractmethod
    def integrate(self, model, X, T, t_m, tail_ratio=1.2):
        pass


class RandomMonteCarlo(EnergyIntegrator):
    r"""Integrator based on Monte-Carlo integration.

    Implements Monte-Carlo integration based on uniform sampling of the 
    integration interval.

    Parameters
    ----------
    num_samples : int
        Number of random samples in the integration
    """
    def __init__(self, num_samples: int):
        super(RandomMonteCarlo, self).__init__()
        self.N = num_samples

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):

        return torch.log(self.integrate(model, X, T, t_m, tail_ratio=1.2))

    def integrate(self, model, X, T, t_m, tail_ratio=1.2):

        X_tail = torch.cat((X, T.reshape(-1, 1)*0+tail_ratio*t_m), dim=1)
        F_tail = torch.exp(model.energy(X_tail)).reshape(-1, 1)

        T = T.reshape(-1, 1)
        if X.dim() == 1:
            X = X.reshape(-1, 1)

        dt = torch.kron(t_m-T, torch.ones((self.N, 1)))
        t = torch.rand((self.N*X.shape[0], 1))

        X = torch.cat((torch.kron(X, torch.ones((self.N, 1))),
                       dt.reshape(-1, 1)*t.reshape(-1, 1)
                       + torch.kron(T.reshape(-1, 1), torch.ones((self.N, 1)))
                       ), dim=1)

        F = torch.exp(model.energy(X)).reshape(-1, self.N)

        Z = F.sum(axis=1).reshape(-1, 1)/self.N*(t_m-T) \
            + F_tail*(tail_ratio-1)*t_m

        return Z


class EquidistantTrapezoidal(EnergyIntegrator):
    r"""Integrator based on the Trapezoidal rule.

    Implements the Trapezoidal rule on an equidistant grid.

    Parameters
    ----------

    num_points : int
        Number of points in the integration.
    """
    def __init__(self, num_points: int):
        super(EquidistantTrapezoidal, self).__init__()
        self.N = num_points

    def _segment(self, X, T, t_m, tail_ratio):
        if X.dim() == 1:
            X = X.reshape(-1, 1)

        u = torch.ones((self.N,))
        u = u / u.sum()
        #        t = torch.cat((torch.zeros(1), u.cumsum(dim=0)))

        t = torch.cat((torch.zeros(1), u.cumsum(
            dim=0), torch.ones(1) * tail_ratio))
        # u = torch.cat((u, torch.ones(1) * (self.tail_ratio - 1.)))
        # FIXME: tail not handled correctly below
        X = torch.cat((X.repeat_interleave(self.N+2,0),
                       torch.kron(t_m - T.reshape(-1, 1), t.reshape(-1, 1))
                       + torch.kron(T.reshape(- 1, 1),
                                    torch.ones_like(t.reshape(-1, 1)))
                       ), dim=1)
        X[self.N + 1::self.N + 2, -1] = tail_ratio*t_m

        return X, u

    def integrate(self, model, X, T, t_m, tail_ratio=1.2):
        X, u = self._segment(X, T, t_m, tail_ratio)

        F = torch.exp(model.energy(X)).reshape(-1, self.N + 2)

        Z = ((F[:, :-2] @ u + F[:, 1:-1] @ u) / 2).reshape(-1, 1) \
            * (t_m - T.reshape(- 1, 1)).reshape(-1, 1) \
            + F[:, -1].reshape(-1, 1) * t_m * (tail_ratio - 1)

        return Z

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):
        X, u = self._segment(X, T, t_m, tail_ratio)

        f = model.energy(X).reshape(-1, self.N + 2)
        f_ast = f.max(dim=1).values
        f -= f_ast[:, None]
        F = torch.exp(f)

        log_Z = f_ast.reshape(-1, 1) \
            + torch.log(((F[:, :-2] @ u
                          + F[:, 1:-1] @ u) / 2).reshape(- 1, 1)
                        * (t_m - T.reshape(-1, 1)).reshape(- 1, 1)
                        + F[:, -1].reshape(-1, 1) * t_m * (tail_ratio - 1))

        return log_Z


# Simpson coefficients
_W_I = (1/6, 4/6, 1/6)
# Simpson-Trapetsoidal error coefficients
_W_e = (1/12, -1/6, 1/12)

class AdaptiveTrapezoidalSimpsons(EnergyIntegrator):
    r"""Adaptive integrator combining Trapezoidal and Simpson's rule.

    Implements an adaptive integration scheme where the difference between
    the Trapezoidal and Simpson's rule on each segment is used to estimate
    the error on each segment. The method iteratively bisects each interval
    that does not satisfy at least one of the tolerances, starting with the 
    initial integration interval.

    Parameters
    ----------
    
    rtol : float>0, default=1e-4
        Relative tolerance (satisfied if error < value*rtol).
    atol : float>0, default=1e-5
        Absolute tolerance (satisfied if error < rtol).
    """
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-5):
        super(AdaptiveTrapezoidalSimpsons, self).__init__()

        self.rtol = rtol
        self.atol = atol

        # Simpson coefficients
        self.W_I = torch.tensor(_W_I)
        # Simpson-Trapetsoidal error coefficients
        self.W_e = torch.tensor(_W_e)

    def integrate(self, model, X, a, b, tail_ratio=1.2):
        a = a.reshape(-1, 1)
        b = torch.ones_like(a)*b

        rtol_pu = self.rtol/(b-a)
        atol_pu = self.atol/(b-a)

        if X.dim() == 1:
            X = X.reshape(-1, 1)

        X_ = torch.cat((X, b*tail_ratio), dim=1)

        # Contribution from tail
        Z = torch.exp(model.energy(X_)) * b * (tail_ratio - 1)

        def f(t, inds=slice(None)):
            X_[inds, -1] = t.view(-1,)
            return torch.exp(model.energy(X_[inds]))

        fl = f(a)
        fh = f(b)
        N = len(fl)

        not_evaluated = [((a, b), (fl, fh), torch.arange(0, N))]

        while not_evaluated:
            intervall = not_evaluated.pop()
            a, b = intervall[0]

            fl, fh = intervall[1]
            inds = intervall[2]

            while True:
                m = (a+b)/2
                fm = f(m, inds)
                dx = (b-a)

                f_stack = torch.concat((fl, fm, fh), dim=1)

                I_intervall = dx*(f_stack@self.W_I).reshape(-1, 1)
                e = torch.abs(dx**2*(f_stack@self.W_e).reshape(-1, 1))
                accurate = ((e < atol_pu[inds]) | (
                    e < I_intervall[0]*rtol_pu[inds])).reshape(-1,)

                if not torch.all(accurate):

                    Z[inds[accurate]] += I_intervall[accurate]
                    not_evaluated.append(((m[~accurate], b[~accurate]),
                                          (fm[~accurate], fh[~accurate]),
                                          inds[~accurate]))

                    inds = inds[~accurate]
                    b = m[~accurate]
                    a = a[~accurate]
                    fl = fl[~accurate]
                    fh = fm[~accurate]
                else:
                    Z[inds] += I_intervall
                    break

        return Z

    def log_integrate(self, model, X, T, t_m, tail_ratio=1.2):
        return torch.log(self.integrate(model, X, T, t_m, tail_ratio))
