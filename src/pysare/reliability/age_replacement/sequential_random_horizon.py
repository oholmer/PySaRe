# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from collections import namedtuple
import math
import scipy

# %%
# t = policy.t
# r = policy.r
# r[r==0] = .6


def plot_policy(h, r, ax, color=None, tol=2, resolution=np.inf, plot_opt='-', inf_level=None):

    r = r.copy()

    if any(r > 0):
        r_min = r[r > 0].min()
    else:
        r_min = 0

    if inf_level is None:
        r_inf = r.max() + (r.max()-r_min)*.2
    else:
        r_inf = inf_level
    r[r == 0] = r_inf

    # downsample if enough points
    if len(h) > 2*resolution:
        t_min = min(h[r != np.inf])
        t_max = max(h[r != np.inf])

        samples_per_time_unit = resolution/(t_max-t_min)
    else:
        samples_per_time_unit = False

    def find_discontinuities(t, r, tol=2):
        dr_tol = t[1]-t[0]

        dr = abs(np.diff(r, prepend=r[0])).clip(dr_tol)
        dr_windowed = np.min((dr[:-2], dr[1:-1], dr[2:]), axis=0)
        dr_windowed = np.insert(dr_windowed, (0, len(
            dr_windowed)), (dr_windowed[0], dr_windowed[-1]), axis=0)

        return np.where(dr/dr_windowed > tol)[0]

    def find_pieces(t, r, tol=2):
        disc = find_discontinuities(t, r, tol)

        return np.insert(disc, (0, len(disc)), (0, len(t)-1), axis=0)

    pieces = find_pieces(h, r, tol)

    if color is None:
        line, = ax.plot(np.nan, np.nan)
        color = line.get_color()

    for n in range(len(pieces)-1):
        start = pieces[n]
        stop = pieces[n+1]

        if samples_per_time_unit:
            inds = np.round(np.linspace(
                start, stop-1, int(np.ceil((h[stop]-h[start])*samples_per_time_unit)+1))).astype(int)
            ax.plot(h[inds], r[inds], plot_opt, color=color)
        else:
            ax.plot(h[start:stop], r[start:stop], plot_opt, color=color)

    # Wierd code to get the correct yticklabels
    labels = [item.get_text() for item in ax.get_yticklabels()
              if item.get_position()[1] < r_inf]
    ticks = [item.get_position()[1] for item in ax.get_yticklabels()
             if item.get_position()[1] < r_inf]

    ticks += [r_inf]
    labels += ['$\infty$']
    # ax.set_yticks([tick for tick in ax.get_yticks() if tick < r_inf] + [r_inf])
    ax.set_yticks(ticks)

    print(labels)
    print(ticks)
    # labels = [item.get_text() for item in ax.get_yticklabels()]
    # labels += ['$\infty$']
    ax.set_yticklabels(labels)
    # plot_policy(t,r,plt,tol=1.1)

# def plot_policy(t, r, ax, color=None, tol=2):

#     r = r.copy()

#     def find_discontinuities(t, r, tol = 2):
#         dr_tol = t[1]-t[0]


#         dr = abs(np.diff(r,prepend=r[0])).clip(dr_tol)
#         dr_windowed = np.min((dr[:-2],dr[1:-1],dr[2:]),axis=0)
#         dr_windowed = np.insert(dr_windowed,(0, len(dr_windowed)),(dr_windowed[0], dr_windowed[-1]),axis = 0)

#         return np.where(dr/dr_windowed>tol)[0]

#     def find_pieces(t,r,tol=2):
#         disc = find_discontinuities(t,r,tol)

#         return np.insert(disc,(0, len(disc)),(0, len(t)-1),axis = 0)

#     pieces = find_pieces(t,r)

#     if color is None:
#         line, = ax.plot(np.nan,np.nan)
#         color = line.get_color()


#     if any(r>0):
#         r_min = r[r>0].min()
#     else:
#         r_min = 0

#     r_inf = r.max() + (r.max()-r_min)*.2
#     r[r==0] = r_inf


#     for n in range(len(pieces)-1):
#         start = pieces[n]
#         stop = pieces[n+1]

#         ax.plot(t[start:stop],r[start:stop],color=color)


#     # Wierd code to get the correct yticklabels
#     labels = [item.get_text() for item in ax.get_yticklabels() if item.get_position()[1]<r_inf]
#     ticks = [item.get_position()[1] for item in ax.get_yticklabels() if item.get_position()[1]<r_inf]

#     ticks += [r_inf]
#     labels +=  ['$\infty$']
#     # ax.set_yticks([tick for tick in ax.get_yticks() if tick < r_inf] + [r_inf])
#     ax.set_yticks(ticks)

#     print(labels)
#     print(ticks)
#     # labels = [item.get_text() for item in ax.get_yticklabels()]
#     # labels += ['$\infty$']
#     ax.set_yticklabels(labels)
#     # plot_policy(t,r,plt,tol=1.1)

def renewal_function(S, t_m, N):
    t = np.linspace(0, t_m, N)
    m = np.zeros_like(t)

    F = 1 - S(t)
    dF = np.diff(F)

    for n in range(2, N):
        m[n] = F[n] + np.sum(m[n - 1:: -1] * dF[:n])

    return pd.DataFrame({"t": t, "m": m}).set_index("t")


def calculate_random_horizon_age_replacement(S_f, S_h, c_f, c_p, t_m, N):
    def S_lim(t):
        Sm = S_h(t_m)
        return (S_h(t) - Sm) / (1 - Sm)

    def Sht(t, t0):
        if t0 >= t_m:
            return t * 0
        else:
            return S_lim(t + t0) / S_lim(np.array([t0]))

        S = S_h(t + t0) / S_h(t0)
        S[t0 >= tm] = 0
        return S

    t = np.linspace(0, t_m, N)

    C = np.zeros_like(t)
    r = np.zeros_like(C)

    # C[-1] = 1.0 # HERE----------------------------------------------------

    for n in range(N - 2, -1, -1):
        km = N - n - 1

        Shn = Sht(t[: km], t[n])
        Sfn = S_f(t[1: km + 1])

        dSf = S_f(t[:km]) - S_f(t[1: km + 1])

        Cnk = C[n + 1:]

        Cnf = np.cumsum((Cnk + c_f) * Shn * dSf)
        Cnp = (c_p + Cnk) * Shn * Sfn
        Cn = Cnf + Cnp

        ind = np.argmin(Cn)
        C[n] = Cn[ind]
        r[n] = ind + 1

        if Cnf[-1] <= C[n]:
            C[n] = Cnf[-1]
            r[n] = km

        if r[n] == km:
            r[n] = 0
    return pd.DataFrame({"t": t, "C": C, "r": r, "S_lim": S_lim(t)}).set_index("t")


def random_horizon_bounds(S_f, S_h, c_f, c_p, t_m, N):
    def Sht(t, t0):
        if t0 >= t_m:
            return t * 0
        else:
            return S_h(t+t0) / S_h(np.array([t0]))

    t = np.linspace(0, t_m, N)

    m = renewal_function(S_f, t_m, N)["m"].values[:-1]

    c_ub = np.zeros(N)
    for n in range(N):
        dF = np.diff(-Sht(t, t[n]))
        c_ub[n] = np.sum(dF * m)

    return pd.DataFrame({"ub": c_ub * c_f, "lb": c_ub * c_p, "t": t}).set_index("t")


def finit_support_bound(S_f, S_h, c_f, c_p, t_0, t_m, N_0):

    def Sht(t, t0):
        if t0 >= t_m:
            return t * 0
        else:
            return S_h(t+t0) / S_h(np.array([t0]))

    dt = t_0/(N_0-1)
    t = np.arange(0, t_m+dt/2, dt)

    m = renewal_function(S_f, t[-1], len(t))['m'].values

    B = np.zeros(N_0)

    for n in range(N_0):

        dF = np.diff(-S_h(t[N_0:])/S_h(t[n]))
        B[n] = np.sum(dF*m[N_0-n:N_0-n+len(dF)])

    return pd.DataFrame({"B": c_f*B, "L": c_p*B, "t": t[:N_0]}).set_index("t")


def finit_support_ractive(S_f, S_h, c_f, t_0, t_m, N_0):

    def Sht(t, t0):
        if t0 >= t_m:
            return t * 0
        else:
            return S_h(t+t0) / S_h(np.array([t0]))

    dt = t_0/(N_0-1)
    t = np.arange(0, t_m+dt/2, dt)

    m = renewal_function(S_f, t[-1], len(t))['m'].values

    B = np.zeros(N_0)

    for n in range(N_0):

        dF = np.diff(-S_h(t[n:N_0])/S_h(t[n]))
        B[n] = c_f*np.sum(dF*m[:len(dF)])

    return pd.DataFrame({"B": B, "t": t[:N_0]}).set_index("t")


Problem = namedtuple("Problem", "S_h S_f c_p c_f")


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def planned_replacements(t, r):

    N = np.zeros_like(t)

    if r[-1] > 0:
        N[-1] = 1

    for n in reversed(range(len(t)-1)):

        if r[n] > 0:
            N[n] = 1 + N[find_nearest(t, t[n]+r[n])]
        else:
            N[n] = 0

    return N


class TimeBasedSequentialPolicy:
    def __init__(self, t, r, C, ub, lb, S_h, S_f) -> None:
        self.t = t
        self.r = r
        self.C = C
        self.ub = ub
        self.lb = lb
        self.S_h = S_h
        self.S_f = S_f

        self.num_planed_replacements = planned_replacements(t, r)

    def replacement_age(self, t):
        if t <= self.t[-1]:
            ind = find_nearest(self.t, t)
            if self.r[ind] > 0:
                return self.r[ind]

        return np.inf

    def plot_policy(self, ax, color=None, tol=2, resolution=np.inf, plot_opt='-', inf_level=None):
        return plot_policy(self.t, self.r, ax, color, tol, resolution, plot_opt, inf_level)


def sequential_random_horizon(failure_distribution, horizon_distribution, c_p, c_f, finite_support_approximation, num_disc_points, min_bound_approx_time):

    policy = calculate_random_horizon_age_replacement(
        failure_distribution.survival_probability, horizon_distribution.survival_probability,  c_f, c_p, finite_support_approximation, num_disc_points)

    bounds = finit_support_bound(failure_distribution.survival_probability, horizon_distribution.survival_probability,
                                 c_f, c_p, finite_support_approximation, finite_support_approximation+min_bound_approx_time, num_disc_points)

    t = policy.index.values
    S0 = horizon_distribution.survival_probability(
        finite_support_approximation)
    St = horizon_distribution.survival_probability(t)

    return TimeBasedSequentialPolicy(t,
                                     policy['r'].values*(t[1]-t[0]),
                                     policy['C'].values,
                                     policy['C'].values*(1-S0/St)+bounds['B'],
                                     np.max((policy['C'].values*(1-S0/St) +
                                             bounds['L'], policy['C']), axis=0),
                                     horizon_distribution.survival_probability(t), failure_distribution.survival_probability(t))


def simulate_single(Th, Tf, c_p, c_f, policy):
    th = Th()

    tc = 0
    c = 0
    while True:
        tf = Tf()
        rn = policy.replacement_age(tc)

        if tf < rn:
            tc += tf
            if tc >= th:
                break
            else:
                c += c_f

        else:
            tc += rn
            if tc >= th:
                break
            else:
                c += c_p

    return c


def simulate(horizon_distribution, failure_distribution, c_p, c_f, policy, N):
    C = np.zeros(N)

    for n in range(N):
        C[n] = simulate_single(horizon_distribution.simulate,
                               failure_distribution.simulate, c_p, c_f, policy)

    return C

# %%
