import numpy as np
import matplotlib.pyplot as plt
import math


def plot_policy(h, r, ax, color=None, tol=2, resolution=np.inf, plot_opt='-', inf_level=None):

    r = r.copy()


    r[r-h>=0]=0

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
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


class HorizonBasedSequentialPolicy:
    def __init__(self, h, r, C, S) -> None:
        self.h = h
        self.r = r
        self.expected_cost = C
        self.survival_function = S

    def replacement_age(self, h):
        if h <= self.h[-1]:
            ind = find_nearest(self.h, h)
            if self.r[ind] > 0:
                return self.r[ind]

        return np.inf

    def plot_policy(self, ax, color=None, tol=2, resolution=np.inf, plot_opt='-', inf_level=None):
        return plot_policy(self.h, self.r, ax, color, tol, resolution, plot_opt, inf_level)

    def plot(self,ax=None, resolution=np.inf):
      
        # Every nth index so that in total *resolution* are plotted
        inds = self.h<0
        inds[::int(max(1,len(self.h)//resolution))] = True
        inds[-1] = True
      
        if ax is None:
            fig, ax = plt.subplots(3, 1, num=1, clear=True)
        else:
            fig=ax[0].figure
        ax[0].plot(self.h[inds],self.survival_function[inds])
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Surv. func.')
        ax[1].plot(self.h[inds], self.expected_cost[inds])
        ax[1].set_ylabel('Cost')

        self.plot_policy(ax[2], resolution=resolution)
        fig.tight_layout()
        ax[2].set_ylabel('Policy')
        ax[2].set_xlabel('Horizon')
        return ax


def _estimate_density(S, f_in, dt):
    f = np.zeros_like(S)

    if f_in is None:
        f[0] = (S[0]-S[1])/dt
    else:
        f[0] = f_in

    for n in range(1, len(S)):
        f[n] = 2*(S[n-1]-S[n])/dt - f[n-1]

    return f


def _sequential_finite_horizon(S, h, c_p, c_f, f=None,):

    dt = h[1]-h[0]
    C = np.zeros_like(S)
    r = np.zeros_like(S)

    # If f not provided
    if not isinstance(f, np.ndarray):
        f = _estimate_density(S, f, dt)

    # Main loop
    for n in range(1, len(S)):
        Ck = np.zeros((n,))

        JC = np.flip(C[:n+1])

        l = (c_f + JC)*f[:n+1]
        L = np.cumsum(l[1:]+l[:-1])*dt/2

        Ck = (c_p+JC[1:])*S[1:n+1] + L
        C_inf = L[-1]

        min_ind = Ck.argmin()

        if Ck[min_ind] < C_inf:
            r[n] = (min_ind+1)*dt
            C[n] = Ck[min_ind]
        else:
            r[n] = (n+1)*dt
            C[n] = C_inf

    return HorizonBasedSequentialPolicy(h, r, C, S)


def sequential_finite_horizon(failure_distribution, c_p, c_f, horizon, num_disc_points):

    if hasattr(failure_distribution, 'to_numpy'):
        failure_distribution = failure_distribution.to_numpy()

    h = np.linspace(0.0, horizon, num_disc_points)

    S = np.zeros_like(h)
    S[h <= failure_distribution.max_time()] = failure_distribution.survival_function(
        h[h <= failure_distribution.max_time()])

    return _sequential_finite_horizon(S, h, c_p, c_f)

    


def _adaptive_age_replacement(Sc, dt, c_p, c_f, S=None, C=None, fc=None, f=None, interpolate=True):

    if S is not None:
        r, C = age_replacement(S, dt, c_p, c_f, f)

    C = C[:len(Sc)]

    if not isinstance(fc, np.ndarray):
        fc = estimate_density(Sc, fc, dt)

    n = len(Sc)-1

    Ck = np.zeros((n,))

    JC = np.flip(C)

    l = (c_f + JC)*fc
    L = np.cumsum(l[1:]+l[:-1])*dt/2

    Ck = (c_p+JC[1:])*Sc[1:] + L
    C_inf = L[-1]

    min_ind = Ck.argmin()

    if Ck[min_ind] < C_inf:
        rc = (min_ind+1)*dt
        Cc = Ck[min_ind]

        if interpolate:
            if min_ind > 0 and min_ind < n-1:
                x = [(min_ind)*dt, (min_ind+1)*dt, (min_ind+2)*dt]
                rc, Cc = quadratic_solution(x, Ck[min_ind-1:min_ind+2])

    else:
        rc = (n+1)*dt
        Cc = C_inf

    return rc, Cc
