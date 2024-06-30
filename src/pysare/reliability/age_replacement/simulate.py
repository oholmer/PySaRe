import numpy
import pandas


def _simulate_finite_horizon(failure_distribution, horizon, policy):
    N_f = 0
    N_p = 0

    h = horizon

    while True:

        T = failure_distribution.sample()

        r = policy.replacement_age(h)
        if T < r:
            h -= T
            if h <= 0.0:
                return N_p, N_f
            N_f += 1
        else:
            h -= r
            if h <= 0.0:
                return N_p, N_f
            N_p += 1


def simulate(failure_distribution, horizon, c_p, c_f, policy, num_simulations=1):
    N_p = numpy.zeros(num_simulations)
    N_f = numpy.zeros(num_simulations)

    try:
        failure_distribution.sample()
    except Exception as error:
        raise Exception("Was no able to sample failure_distribuiton. "
                    "Try converting it to a spline  approximation using"
                    " to_spline_approximation.") from error
        

    for n in range(num_simulations):
        N_p[n], N_f[n] = _simulate_finite_horizon(
            failure_distribution, horizon, policy)

    return pandas.DataFrame({'num_failures': N_f, 'num_preventive': N_p, 'cost': N_f*c_f+N_p*c_p})
