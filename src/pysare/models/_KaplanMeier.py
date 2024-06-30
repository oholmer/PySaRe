import numpy as np
import torch

class KaplanMeierEstimator():
    def __init__(self, T, E=None) -> None:

        T = np.array(T)
        order = T.argsort()
        T = T[order]

        if E is None:
            E = np.ones_like(T, dtype=bool)
        else:
            E = np.array(E, dtype=bool)[order]

        first_unique_ind = np.where(
            np.diff(T, prepend=T[0]-1.0, append=T[-1]+1.0) > 0)[0]
        num_prev_failures = np.zeros(len(T)+1, dtype=int)
        np.cumsum(E, out=num_prev_failures[1:])

        num_failures = np.diff(num_prev_failures[first_unique_ind])
        num_at_risc = len(T)-first_unique_ind[:-1]

        have_failure = num_failures > 0
        step_inds = first_unique_ind[:-1][have_failure]

        num_failures = num_failures[have_failure]
        num_at_risc = num_at_risc[have_failure]


        if T[step_inds[-1]]==T[-1]:
            self.times = T[step_inds]
            Q = (1-num_failures/num_at_risc)
            self.survival_probabilities = np.cumprod(Q)
        else:
            self.times = np.empty(len(step_inds)+1)
            self.times[:-1] = T[step_inds]
            self.times[-1] = T[-1]
            self.survival_probabilities = np.empty_like(self.times)
            self.survival_probabilities[-1] = 0.0
            Q = (1-num_failures/num_at_risc)
            self.survival_probabilities[:-1] = np.cumprod(Q) 

        self.probabilities = np.diff(self.survival_probabilities, prepend=1.0)

    def survival_function(self, T):
        T = np.array(T)
        inds = np.atleast_1d(np.searchsorted(self.times,T,side="right")-1)
        
        probs = self.survival_probabilities[inds]
        probs[inds<0] = 1.0
        return probs.reshape(T.shape)
    
    def sample(self, size=None, generator=None):
        
        u = torch.rand(size=tuple(np.atleast_1d(size)),
                   dtype=torch.float64,
                   generator=generator).numpy()

        inds = (self.survival_probabilities.shape[0] - 
                np.searchsorted(np.flip(self.survival_probabilities), u))
        
        return self.times[inds].reshape(size)