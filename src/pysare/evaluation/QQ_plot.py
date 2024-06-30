import lifelines
import torch
import numpy as np

def QQ_plot(model, dataloader, num_times=np.inf, model_opts='--+'):

    # Full vector of times  and events
    # ===========================================================================
    T = []
    E = []

    for X_batch, T_batch, E_batch in dataloader:
        T.append(T_batch.reshape(-1,))
        E.append(E_batch.reshape(-1,))

    if torch.is_tensor(T[0]):
        T = torch.concat(T)
        E = torch.concat(E)
        T_np = np.array(T.cpu())
        E_np = np.array(E.cpu())
    else:
        T = np.concatenate(T)
        E = np.concatenate(E)
        T_np = T
        E_np = E

    def find_nearest(array, values):
        inds = np.searchsorted(array, values).clip(1)
        inds[ values - array[inds-1] < array[inds] - values] -= 1
        return inds

    order = np.argsort(T_np)
    T, E, T_np, E_np = T[order], E[order], T_np[order], E_np[order]



    T_event, T_event_np = T[E], T_np[E_np]

    if num_times < len(T_event):
        t = np.linspace(T_event_np[0], T_event_np[-1], num_times)

        inds = np.unique(find_nearest(T_event_np, t))
        T_event, T_event_np = T_event[inds], T_event_np[inds]


    KM = lifelines.KaplanMeierFitter().fit(T_np, E_np)
    ax = KM.plot()


    S_np = np.empty_like(T_event_np)

    with torch.no_grad():
        for n in range(T_event_np.shape[0]):

            R = []
            for X_batch, _, _ in dataloader:
                R.append(np.array(model.survival_function(
                    X_batch, T_event[n:n+1].repeat(X_batch.shape[0])).reshape(-1,).cpu(), copy=False))
            R = np.concatenate(R)

            S_np[n] = R.mean()

    ax.plot(T_event_np, S_np,model_opts, label='Model mean over covariates')
    ax.legend()

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['Kaplan-Meier estimate', 'Model mean over covariates'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival function')
