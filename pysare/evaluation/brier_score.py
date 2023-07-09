import torch.utils.data
import pysare
import numpy as np
import matplotlib.pyplot as plt
import lifelines as lf
import pandas as pd

def brier_score(model, dataset, t=None, num_t=None, censoring_distribution="KM",
                ax=None):
    
    T = dataset.T.numpy()
    E = dataset.E.numpy()
    X = dataset.X.numpy()

    if t is None:
        t = np.linspace(0, T.max(), num_t)

    if censoring_distribution == "KM":
        KMC = lf.KaplanMeierFitter()
        KMC.fit(T, ~E)
        S_cens_t = KMC.survival_function_at_times(t).values
        S_cens_T_inv = 1/KMC.survival_function_at_times(T.reshape(-1,)).values.reshape(-1,)
    elif censoring_distribution is False:
        S_cens_t = np.ones_like(t)
        S_cens_T_inv = np.ones((T.shape[0],))
    else:
        S_cens_t = censoring_distribution(t)
        S_cens_T_inv = censoring_distribution(T)

    B = np.zeros_like(t)

    for n in range(t.shape[0]):

        R = model.survival_probability(X, t[n]).reshape(-1,)


        B1 = ((R**2)*(T <= t[n])*S_cens_T_inv)[E]
        
        B[n] = (B1.sum() + ((T > t[n])*(1 - R)**2).sum()/S_cens_t[n])/T.shape[0] 
    
    # Plot if ax is not False
    if not ax is False:
        if ax is None:
            if not censoring_distribution is False:
                fig, ax = plt.subplots(nrows=2)
            else:
                fig, ax = plt.subplots(nrows=1)

        if not censoring_distribution is False:
            if censoring_distribution == 'KM':
                KMC.plot_survival_function(ax=ax[1])
            else:
                ax[1].plot(t, S_cens_t)
            ax[1].legend(['Kaplan-Meier Estimate','Confidence Intervall (95%)'])
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel('Censoring Distribution')
        else:
            ax = [ax]
        ax[0].plot(t, B)
        ax[0].set_ylabel('Brier Score')
        ax[0].set_xlabel('Time')
        plt.tight_layout()

    return pd.DataFrame(data={'score': B, 'time': t, 'censoring_distribution': S_cens_t})
    

def integrated_brier_score(model, dataset, t=None, num_t=None, censoring_distribution="KM",
                ax=None):
    
    if not ax is False:
        if ax is None:
            if not censoring_distribution is False:
                fig, ax = plt.subplots(nrows=2)
            else:
                fig, ax = plt.subplots(nrows=1)
    
    if not censoring_distribution is False:
        brier = brier_score(model, dataset, t, num_t, censoring_distribution, ax)
    else:
        brier = brier_score(model, dataset, t, num_t, censoring_distribution, ax=False)
        ax = [ax]
    
    B = brier.score.values
    t = brier.time.values
    
    IB =  np.zeros_like(B)
    IB[1:] += (B[1:]+B[:-1])*np.diff(t)/2
    IB = IB.cumsum()/(t-t.min())

    # Plot if ax is not False
    if not ax is False:
        ax[0].clear()
        ax[0].plot(t, IB)
        ax[0].set_ylabel('Integrated Brier Score')
        ax[0].set_xlabel('Time')
        plt.tight_layout()

    return pd.DataFrame(data={'integrated_score': IB, 'time': t, 'censoring_distribution': brier.censoring_distribution.values})