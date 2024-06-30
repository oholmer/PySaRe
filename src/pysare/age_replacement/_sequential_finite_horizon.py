import numpy as np
import matplotlib.pyplot as plt


def midpoints(t, y):
    step_inds = np.where(np.diff(y[1:-1]) != 0)[0]+1
    y = np.insert(0.5*(y[step_inds]+y[step_inds+1]),
                  [0, len(step_inds)], y[[0, -1]])
    t = np.insert(0.5*(t[step_inds]+t[step_inds+1]),
                  [0, len(step_inds)], t[[0, -1]])
    return t, y


def interpolate(t, y, samples_per_tu):

    num_samples = max(np.ceil(samples_per_tu*(t[-1]-t[0])).astype(int), 2)

    if num_samples < len(t):
        t_plot = np.linspace(*t[[0, -1]], num_samples)
        y_plot = np.interp(t_plot, t, y)

        return t_plot, y_plot
    else:
        return t, y


def get_segments(t, y, resolution, tol):
    # Position of midpoint in relation to the adjacent
    x = (t[1:-1]-t[:-2])/(t[2:]-t[:-2])

    isinf = np.isinf(y)
    becomes_inf = np.diff(isinf) != 0

    y_hat = y[2:]*x + (1.0-x)*y[:-2]

    e = (np.abs(y_hat-y[1:-1])-resolution).clip(0.0)/y[1:-1]
    e[np.isnan(e)] = 2*tol
    is_discontinuous = (np.diff((e > tol).astype(
        int), prepend=e[0]) == 1) | becomes_inf[1:]

    disc_inds = np.where(is_discontinuous)[0]+2

    segments = np.insert(disc_inds, [0, len(disc_inds)], [0, len(t)])
    return segments


def find_nearest(data, keys):
    inds = np.atleast_1d(np.searchsorted(
        data, keys, side='left').clip(1, data.shape[0]-1))

    inds[keys-data[inds-1] < data[inds]-keys] -= 1

    return inds.reshape(keys.shape)


class HorizonBasedPolicy():
    def __init__(self,
                 horizon,
                 replacement_age,
                 expected_cost,
                 survival_probability,
                 preventive_cost=None,
                 failure_cost=None,
                 grid_spacing=0.0,
                 interpolation='nearest',
                 ) -> None:
        self.horizon = horizon
        self._replacement_age = replacement_age
        self.expected_cost = expected_cost
        self.survival_probability = survival_probability
        self.preventive_cost = preventive_cost
        self.failure_cost = failure_cost
        self.grid_spacing = grid_spacing
        self.interpolation = interpolation

    def __getitem__(self, horizons):

        horizons = np.array(horizons, dtype=float)
        horizons[horizons < 0] = np.nan
        horizons[horizons > self.horizon[-1]+0.5*self.grid_spacing] = np.nan

        if self.interpolation == 'nearest':
            inds = find_nearest(self.horizon, horizons)
            replacement_ages = self._replacement_age[inds]
        if self.interpolation == 'linear':
            replacement_ages = np.interp(
                horizons, self.horizon, self._replacement_age)

        replacement_ages = np.atleast_1d(replacement_ages)
        replacement_ages[replacement_ages == 0] = self.horizon[-1]
        return replacement_ages.reshape(horizons.shape)

    def evaluate(self,
                 failure_distribution,
                 preventive_cost=None,
                 failure_cost=None):
        if preventive_cost is None:
            preventive_cost = self.preventive_cost

        if failure_cost is None:
            failure_cost = self.failure_cost

        return _evaluate_sequential_finite_horizon(
            failure_distribution.survival_function(self.horizon),
            self.horizon,
            self._replacement_age,
            preventive_cost,
            failure_cost)

    def simulate(self,
                 failure_distribution,
                 num_simulations=1,
                 preventive_cost=None,
                 failure_cost=None,
                 horizon=None
                 ):
        if preventive_cost is None:
            if self.preventive_cost is None:
                raise ValueError(
                    'Preventive cost not set for policy and must be provided')
            preventive_cost = self.preventive_cost

        if failure_cost is None:
            if self.failure_cost is None:
                raise ValueError(
                    'Failure cost not set for policy and must be provided')
            failure_cost = self.failure_cost

        if horizon is None:
            horizon = self.horizon[-1]

        remaining_horizon = np.full((num_simulations,), horizon)
        num_preventive = np.zeros_like(remaining_horizon, dtype=int)
        num_failures = np.zeros_like(remaining_horizon, dtype=int)

        active = np.arange(0, num_simulations)

        while len(active) > 0:
            failure_time = failure_distribution.sample(
                len(active)).reshape(active.shape)

            replacement_age = self[remaining_horizon[active]]

            is_preventive = failure_time > replacement_age
            is_failure = np.logical_not(is_preventive)

            replacement_time = failure_time
            replacement_time[is_preventive] = replacement_age[is_preventive]

            remaining_horizon[active] -= replacement_time

            finished = remaining_horizon[active] <= 0

            num_preventive[active[~finished & is_preventive]] += 1
            num_failures[active[~finished & is_failure]] += 1

            active = active[~finished]

        num_preventive = num_preventive.squeeze()
        num_failures = num_failures.squeeze()

        cost = preventive_cost*num_preventive + failure_cost*num_failures
        return {'cost': cost,
                'num_preventive': num_preventive,
                'num_failures': num_failures}

    def plot_policy(self,
                    ax=None,
                    color=None,
                    linewidth=None,
                    linestyle=None,
                    draw_inf=True,
                    resolution=None,
                    use_midpoints=True,
                    discontinuity_tolerance=1e-6):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        if resolution is not None:
            samples_per_tu = resolution/(self.horizon[-1]-self.horizon[0])
        else:
            samples_per_tu = np.inf

        h = self.horizon
        r = self._replacement_age.copy()
        is_inf = (r == 0.0) | (r >= h)

        if not is_inf.all():
            r_max = np.nanmax(r[~is_inf])
        else:
            r_max = 0.0
            
        # Something big to make sure it becomes a segment
        r[is_inf] = np.inf
        segments = get_segments(
            h, r, self.grid_spacing, discontinuity_tolerance)

        if isinstance(draw_inf, bool):
            r[is_inf] = r_max*1
        else:
            r[is_inf] = float(draw_inf)

        def plot_segment(n, color):

            if is_inf[segments[n]]:

                if draw_inf is not False:
                    inds = [segments[n], segments[n+1]-1]
                    line, = ax.plot(h[inds], r[inds], '--',
                                    color=color, linewidth=linewidth)
                else:
                    return color
            else:
                inds = slice(segments[n], segments[n+1])
                if use_midpoints:
                    hn, rn = midpoints(h[inds], r[inds])
                else:
                    hn, rn = h[inds], r[inds]

                if samples_per_tu < np.inf:
                    hn, rn = interpolate(hn, rn, samples_per_tu)

                line, = ax.plot(hn, rn, color=color, linewidth=linewidth, linestyle=linestyle)

            if color is None:
                return line.get_color()
            else: 
                return color

        color = plot_segment(0, color)

        for n in range(1, len(segments)-1):
            color = plot_segment(n, color)
        return ax, segments


def _evaluate_sequential_finite_horizon(S, h, r, c_p, c_f):
    C = np.zeros_like(S)
    # r = np.zeros_like(S)
    N = len(C)
    dF = -np.diff(S)
    dh = h[1]-h[0]

    for n in range(1, N):
        if (r[n] <= 0) or (r[n] >= n*dh):
            JC = np.flip(C[:n])
            V = np.sum((c_f + JC)*dF[:n])
            C[n] = V
        else:
            k = max(1, int(np.round(r[n]/dh))-1)

            JC = np.flip(C[n-k-1:n])
            V = np.sum((c_f + JC)*dF[:k+1])
            W = (c_p+JC[-1])*S[k] + V
            C[n] = W

    return HorizonBasedPolicy(h, r, C, S, grid_spacing=dh, preventive_cost=c_p, failure_cost=c_f)


def _sequential_finite_horizon(S, h, c_p, c_f):
    C = np.zeros_like(S)
    r = np.zeros_like(S)
    N = len(C)
    dF = -np.diff(S)
    dh = h[1]-h[0]

    for n in range(1, N):
        JC = np.flip(C[:n])
        V = np.cumsum((c_f + JC)*dF[:n])
        W = (c_p+JC)*S[:n] + V

        k_opt = W.argmin()
        W_opt = W[k_opt]

        if W_opt < V[-1]:


            ind = np.where(W-W_opt <= np.spacing(W_opt))[0][-1]
            r[n] = (ind+1)*dh
            C[n] = W_opt
        else:
            r[n] = 0.0
            C[n] = V[-1]

    return HorizonBasedPolicy(h, r, C, S, grid_spacing=dh, preventive_cost=c_p, failure_cost=c_f)


def sequential_finite_horizon(failure_distribution,
                              preventive_cost,
                              failure_cost,
                              horizon,
                              num_disc_points):
    if hasattr(failure_distribution, 'to_numpy'):
        failure_distribution = failure_distribution.to_numpy()

    h = np.linspace(0.0, horizon, num_disc_points)

    # S = np.zeros_like(h)
    # S[h <= failure_distribution.max_time()] = failure_distribution.survival_function(
    #     h[h <= failure_distribution.max_time()])
    S = failure_distribution.survival_function(h)

    return _sequential_finite_horizon(S, h, preventive_cost, failure_cost)
