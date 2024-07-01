from __future__ import annotations
import warnings
import pickle
import torch
import numpy as np
from numpy.typing import ArrayLike
from torch._prims_common import DeviceLikeType
from torch.types import _dtype
from typing import Optional, Callable

class Dataset(torch.utils.data.Dataset):
    r"""
    Implements a torch dataset for survival data.

    Attributes
    ----------
    X (Tensor):
        Tensor with features/covariates.
    T (Tensor):
        Tensor with failure times.
    E (Tensor):
        Bool Vector with event types, 1 for failure and 0 for censoring

    Methods
    -------
    right_censor(self, censoring_time: float):
        Right censors dataset.
    save_to_file
        Saves dataset to file.
    load_from_file(cls, filename: str):
        Loads dataset from file.
    split(proportions:Arraylike, sizes: ArrayLike):
        Splits dataset into subsets according to x.
    to(...):
        Performs Tensor dtype and/or device conversion. 
    """

    def __init__(self, X: ArrayLike, T: ArrayLike, E: Optional[ArrayLike] = None, copy: bool = True) -> None:
        r"""
        Constructs a Dataset.

        Parameters
        ----------
            X : (n,...) ArrayLike
                Features/covariates with first dimension as batch dimension.
            T : (n,) ArrayLike
                Recorded times, one-dimensional.
            E : (n,) ArrayLike
                Vector of event types, entries evaluates as true for a recorded 
                event and false for censoring. 
            copy (bool):
                If true (default) parameters X, T, and E will be copied. Otherwise, 
                they will be copied only if they ar not tensors. 
        """
        def get_tensor(tensor):
            if copy:
                return tensor.clone()
            else:
                return tensor

        # Copy or convert X
        if torch.is_tensor(X):

            self.X = get_tensor(X)
                
        else:
            self.X = torch.tensor(X, dtype=torch.get_default_dtype())
        N = self.X.shape[0]

        # Copy or convert T to onedimensional tensor
        if torch.is_tensor(T):
            self.T = get_tensor(T)
        else:
            self.T = torch.tensor(T, dtype=torch.get_default_dtype())
        # Check dimensions
        if len(self.T.shape) > 1:
            warnings.warn("T not onedimensional, continuing by reshaping it.")
            self.T = self.T.reshape(-1,)
        # Check size
        if not self.T.shape[0] == N:
            raise ValueError(
                "Number of elements in T not equal to shape of first dimension of X")

         # Copy or convert E
        if E is None:
            self.E = torch.ones_like(self.T, dtype=torch.bool)
        else:
            if torch.is_tensor(E):
                self.E = get_tensor(E)
            else:
                self.E = torch.tensor(E)
            # Check type
            if not self.E.dtype == torch.bool:
                warnings.warn("E not bool, continuing by converting it.")
                self.E = self.E.type(torch.bool)
            # Check dimensions
            if len(self.E.shape) > 1:
                warnings.warn(
                    "E not onedimensional, continuing by reshaping it.")
                self.E = self.E.reshape(-1,)
            # Check size
            if not self.E.shape[0] == N:
                raise ValueError(
                    "Number of elements in E not equal to shape of first dimension of X")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.E[idx]

    def right_censor(self, censoring_time: float):
        r"""
        Right censors the dataset.

        Parameters
        ----------
            censoring_time (float):
                Censoring time

        Returns
        ----------
            None
        """
        self.E[self.T > censoring_time] = False
        self.T[self.T > censoring_time] = censoring_time

    def save_to_file(self, filename):
        r"""
        Saves dataset to a file.

        Parameters
        ----------
            filename (str):
                Name of file where dataset will be saved.

        Returns
        ----------
            None
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str) -> Dataset:
        r"""
        Loads a dataset from a file.

        Parameters
        ----------
            filename (str): 
                Name of file from where to load the dataset.

        Returns
        ----------
            Dataset
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def split(self, proportions: ArrayLike = None, sizes = None, generator=None):


        if proportions:
            proportions = np.array(proportions, ndmin=1).reshape(-1,)

            sizes = (proportions*len(self)).round().astype(int)
            if sizes.sum()<len(self):
                sizes = np.concatenate((sizes,[len(self)-sizes.sum()]))
        elif not sizes:
            raise ValueError("Either proportions or sizes must be provided")


        return torch.utils.data.random_split(self, sizes, generator=generator)



        # N = self.__len__()

        # I = np.array(range(N))
        # np.random.shuffle(I)

        # if not isinstance(x, list):
        #     x = [x]

        # s = (np.rint(np.array(x) * N).astype(np.int64))
        # r = N - s.sum()
        # s = s.cumsum()
        # s = list(s)

        # I_s = self._split_ind(I, s)

        # datasets = []
        # for n in range(len(I_s)):
        #     datasets.append(
        #         Dataset(self.X[I_s[n], :], self.T[I_s[n]], self.E[I_s[n]]))

        # return datasets

    @staticmethod
    def _split_ind(X, I):
        Xs = [X[:I[0]]]
        for n in range(1, len(I)):
            Xs.append(X[I[n - 1]:I[n]])
        Xs.append(X[I[-1]:])
        return Xs

    def __repr__(self):
        return (f"PySaRe Dataset with:\n" +
                f"  {self.T.shape[0]} individuals\n" +
                f"  {self.T.shape[0]-self.E.sum()} censored\n" +
                f"  {tuple(self.X[0].shape)} as shape of each individual's features")


    def plot(self):
        import lifelines

        km_event = lifelines.KaplanMeierFitter().fit(self.T.numpy(),self.E.numpy())
        ax = km_event.plot()
        km_cens = lifelines.KaplanMeierFitter().fit(self.T.numpy(),~self.E.numpy())
        km_cens.plot(ax=ax)

        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['Events', 'Censoring'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival function')
        
        
    def to(self, device:Optional[DeviceLikeType] = None, dtype:Optional[_dtype]=None, data_transform:Optional[Callable]=None):
        r"""
        Performs Tensor dtype and/or device conversion. 

        Parameters
        ----------
            device (DeviceLikeType):
                Device to move dataset to. 
            dtype (_dtype):
                dtype to change the dataset to. 
            data_transform (Callable):
                Function that performs the descried conversion of X, T, and E like
                X_new, T_new, E_new = data_transform(X, T, E)
        Returns
        ----------
            Dataset:
                reference to self.
        """
        if data_transform is None:
            args = {arg: val for arg, val in (
                ("device", device), ("dtype", dtype)) if val is not None}

            self.X = self.X.to(**args)
            self.T = self.T.to(**args)

            # dtype of E should no be changed
            args.pop("dtype", None)
            self.E = self.E.to(**args)
            
        else:
            self.X, self.T, self.E = data_transform(self.X, self.T, self.E)

        return self