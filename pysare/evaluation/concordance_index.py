import torch
import numpy as np
import matplotlib.pyplot as plt
import lifelines as lf
import pandas as pd
def concordance_index(model, dataset, ax=None, censoring='KM'):

    X = dataset.X.detach().numpy()
    T = dataset.T.detach().numpy()
    E = dataset.E.detach().numpy()

    # Event times (not censorings)
    T_event = T[E]
    R = model.survival_probability(X, T.reshape(-1,1))

    # Uniqe times between the first and last recorded event 
    tau = np.sort(np.unique(T[(T<=T_event.max()) & (T>= T_event.min())]))

    if censoring == "KM":
        KMC = lf.KaplanMeierFitter()
        KMC.fit(T, ~E)
        C_T = KMC.survival_function_at_times(T.reshape(-1,)).values.reshape(-1,)
        C_tau = KMC.survival_function_at_times(tau.reshape(-1,)).values.reshape(-1,)
    elif censoring is False:
        C_T = np.ones((T.shape[0],))
        C_tau = np.ones((tau.shape[0],))
    else:
        C_T = censoring(T).reshape(-1,)
        C_tau = censoring(tau).reshape(-1,)

    A = np.zeros_like(T)
    B = np.zeros_like(T)
    for n in range(T.shape[0]):

        S_Tn = model.survival_probability(X,T[n])

        comp = (T[n] < T).reshape(-1,1)
        A[n] = ((S_Tn[n] < S_Tn) & comp).sum()/(C_T[n]**2)*E[n]
        B[n] = comp.sum()/(C_T[n]**2)*E[n]

    C = np.zeros_like(tau)

    for n in range(tau.shape[0]):
        ind = T<=tau[n]
        C[n] = A[ind].sum()/B[ind].sum()

    # Plot if ax is not False
    if not ax is False:
        if ax is None:
            if not censoring is False:
                fig, ax = plt.subplots(nrows=2)
            else:
                fig, ax = plt.subplots(nrows=1)

        if not censoring is False:
            if censoring == 'KM':
                KMC.plot_survival_function(ax=ax[1])
            else:
                ax[1].plot(T,C_T)
            ax[1].legend(['Kaplan-Meier Estimate','Confidence Intervall (95%)'])
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Censoring Distribution')
        else:
            ax = [ax]
        ax[0].plot(tau, C)
        ax[0].set_ylabel('Truncated C-index')
        ax[0].set_xlabel('Time')
        plt.tight_layout()

    return pd.DataFrame(data={'C_index': C, 'time': tau, 'censoring_distribution': C_tau})