import torch
import numpy as np
import warnings

class Dataset(torch.utils.data.Dataset):
    """Implements a torch dataset for survival data.
        Args:
            X: Arraylike with features/covariates.
            T: Vectorlike with failure times.
            E: Bool Vectorlike with event types, true for failure and flase 
               for censoring

        Attributes:
            X: Tensor with features/covariates.
            T: Tensor with failure times.
            E: Bool Vector with event types, 1 for failure and 0 for censoring
    """

    def __init__(self, X, T, E):

        # Copy or convert X
        if torch.is_tensor(X):
            self.X = X.clone()
        else:
            self.X = torch.tensor(X, dtype=torch.get_default_dtype())
        N = self.X.shape[0]

        # Copy or convert T to onedimensional tensor 
        if torch.is_tensor(T):
            self.T = T.clone()
        else:
            self.T = torch.tensor(T, dtype=torch.get_default_dtype())
        # Check dimensions
        if len(self.T.shape) > 1:
            warnings.warn("T not onedimensional, continuing by reshaping it.")
            self.T = self.T.reshape(-1,)
        # Check size
        if not self.T.shape[0] == N:
            raise ValueError("Number of elements in T not equal to shape of first dimension of X")

         # Copy or convert E
        if torch.is_tensor(E):
            self.E = E.clone()
        else:
            self.E = torch.tensor(E)
        # Check type
        if not self.E.dtype == torch.bool:
            warnings.warn("E not bool, continuing by converting it.")
            self.E = self.E.type(torch.bool)
        # Check dimensions
        if len(self.E.shape) > 1:
            warnings.warn("E not onedimensional, continuing by reshaping it.")
            self.E = self.E.reshape(-1,)
        # Check size
        if not self.E.shape[0] == N:
            raise ValueError("Number of elements in E not equal to shape of first dimension of X")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.E[idx]
    
    def right_censor(self, t_c):
        self.E[self.T>t_c] = False
        self.T[self.T>t_c] = t_c

    def split(self, x):

        N = self.__len__()

        I = np.array(range(N))
        np.random.shuffle(I)

        if not isinstance(x, list):
            x = [x]

        s = (np.rint(np.array(x) * N).astype(np.int64))
        r = N - s.sum()
        s = s.cumsum()
        s = list(s)

        I_s = self._split_ind(I, s)

        datasets = []
        for n in range(len(I_s)):
            datasets.append(Dataset(self.X[I_s[n], :], self.T[I_s[n]], self.E[I_s[n]]))

        return datasets

    @staticmethod
    def _split_ind(X, I):
        Xs = [X[:I[0]]]
        for n in range(1, len(I)):
            Xs.append(X[I[n - 1]:I[n]])
        Xs.append(X[I[-1]:])
        return Xs
    


class MultipleSnapshotDataset():
    def __init__(self,ID, X, T, E, t_ss, sort=True):
        
        # Copy or convert X
        if torch.is_tensor(X):
            self.X = X.clone()
        else:
            self.X = torch.tensor(X, dtype=torch.get_default_dtype())
        N = self.X.shape[0]

        # Copy or convert T to onedimensional tensor 
        if torch.is_tensor(T):
            self.T = T.clone()
        else:
            self.T = torch.tensor(T, dtype=torch.get_default_dtype())
        # Check dimensions
        if len(self.T.shape) > 1:
            warnings.warn("T not onedimensional, continuing by reshaping it.")
            self.T = self.T.reshape(-1,)
        # Check size
        if not self.T.shape[0] == N:
            raise ValueError("Number of elements in T not equal to shape of first dimension of X")

         # Copy or convert E
        if torch.is_tensor(E):
            self.E = E.clone()
        else:
            self.E = torch.tensor(E)
        # Check type
        if not self.E.dtype == torch.bool:
            warnings.warn("E not bool, continuing by converting it.")
            self.E = self.E.type(torch.bool)
        # Check dimensions
        if len(self.E.shape) > 1:
            warnings.warn("E not onedimensional, continuing by reshaping it.")
            self.E = self.E.reshape(-1,)
        # Check size
        if not self.E.shape[0] == N:
            raise ValueError("Number of elements in E not equal to shape of first dimension of X")
        
        # Copy or convert ID
        if torch.is_tensor(ID):
            self.ID = ID.clone()
        else:
            self.ID = torch.tensor(ID).type(torch.int64)
        # Check type
        if not self.ID.dtype == torch.int64:
            warnings.warn("ID not integer, continuing by converting it.")
            self.ID = self.ID.type(torch.int64)
        # Check dimensions
        if len(self.ID.shape) > 1:
            warnings.warn("ID not onedimensional, continuing by reshaping it.")
            self.ID = self.ID.reshape(-1,)
        # Check size
        if not self.ID.shape[0] == self.T.shape[0]:
            raise ValueError("Number of elements in ID not equal to shape of first dimension of X")
        
        
        # Copy or convert t_ss to onedimensional tensor 
        if torch.is_tensor(t_ss):
            self.t_ss = t_ss.clone()
        else:
            self.t_ss = torch.tensor(t_ss, dtype=torch.get_default_dtype())
        # Check dimensions
        if len(self.t_ss.shape) > 1:
            warnings.warn("t_ss not onedimensional, continuing by reshaping it.")
            self.t_ss = self.t_ss.reshape(-1,)
        # Check size
        if not self.t_ss.shape[0] == self.T.shape[0]:
            raise ValueError("Number of elements in t_ss not equal to shape of first dimension of X")
        

        
       
        
        if sort:
            self.sort()

        # Find first index of each ID, assuming each ID comes in a row (no mix)
        self.start_ind =  torch.cat((torch.tensor([0]).to(self.ID),
                        torch.torch.where(self.ID.diff())[0]+1,
                        torch.tensor([self.ID.shape[0]]).to(self.ID)),
                        0)
        self.t_max = self.t_ss[self.start_ind[1:]-1]

    def arrange_by(self, indices):
        self.t_ss = self.t_ss[indices]
        self.X = self.X[indices]
        self.T = self.T[indices]
        self.E = self.E[indices]
        self.ID = self.ID[indices]    

    def sort(self):
        _, indices = torch.sort(self.t_ss, dim=0, stable=True)
        self.arrange_by(indices)
        _, indices = torch.sort(self.ID, dim=0, stable=True)
        self.arrange_by(indices)

    def to_dataset(self):
        # torch.concatenate((self.t_0.reshape(-1,1), self.X),0), self.T.reshape(-1,)-self.t_0.reshape(-1,), self.E

        return Dataset(torch.cat((self.t_ss.reshape(-1,1),self.X.reshape(self.t_ss.shape[0],-1)),1), 
                       self.T.reshape(-1,)-self.t_ss.reshape(-1,), 
                       self.E)

    def split(self, split_proportions):
        if not isinstance(split_proportions, list):
            split_proportions = [split_proportions]
        split_proportions = torch.tensor(split_proportions)

        if split_proportions.sum()>=1:
            split_proportions /= split_proportions.sum()
            split_proportions = split_proportions[:-1]

        unique_IDs = self.ID.unique()
        num_IDs = unique_IDs.shape[0]


        unique_IDs = unique_IDs[ torch.randperm(num_IDs)]

        split_ind = torch.round(split_proportions.cumsum(0)*num_IDs)
        last_ind = 0

        
        datasets = []
        for ind in split_ind:
            cur_IDs = unique_IDs[int(last_ind):int(ind)]
            cur_inds = torch.isin(self.ID,cur_IDs)
            

            datasets.append(MultipleSnapshotDataset(self.ID[cur_inds],
                                                        self.X[ cur_inds], 
                                                        self.T[cur_inds], 
                                                        self.E[cur_inds],
                                                        self.t_ss[cur_inds]))
            last_ind = ind

        cur_IDs = unique_IDs[int(last_ind):]
        cur_inds = torch.isin(self.ID,cur_IDs)
        datasets.append(MultipleSnapshotDataset(self.ID[cur_inds],
                                                        self.X[ cur_inds], 
                                                        self.T[cur_inds], 
                                                        self.E[cur_inds],
                                                        self.t_ss[cur_inds]))
        
        return datasets


    def interpolate(self,t):
        # Index where t_0>0
        I_g = torch.where(self.t_ss>t)[0]
        if I_g.shape[0]>0:
            # Index where t_0>0 and first for new ID
            I_u = I_g[torch.where(self.ID[I_g].diff()>0)[0]+1]
            I_u = torch.cat((I_g[0].reshape(1,), I_u), 0)

            I_u = I_u[I_u>0]
            I_u = I_u[ self.ID[I_u] == self.ID[I_u-1]]


            delta = ((self.t_ss[I_u]-t)/(self.t_ss[I_u]-self.t_ss[I_u-1])).reshape(-1,1)

            X_t = self.X[I_u-1]*delta + self.X[I_u]*(1-delta)
            
            # return self.t_0[I_u]*0+t, X_t#, self.T[I_u] - t, self.E[I_u]
            
            if len(X_t.shape) == 1:
                X_t = X_t.reshape(-1,1)

            return X_t, self.t_ss[I_u]*0+t, self.T[I_u],  self.E[I_u]
            return torch.cat([(self.t_ss[I_u]*0+t).reshape(-1,1), X_t],1), self.T[I_u] - t, self.E[I_u] ,X_t
        
        return  torch.Tensor(0, (self.X.shape[1] if len(self.X.shape)>1  else 1)).to(self.X),torch.Tensor(0,).to(self.t_ss), torch.Tensor(0,).to(self.T), torch.Tensor(0,).to(self.E)
        return  torch.Tensor(0, (self.X.shape[1] if len(self.X.shape)>1  else 2)).to(self.X), torch.Tensor(0,).to(self.T), torch.Tensor(0,).to(self.E)


    def resample(self, t):
    
        if not getattr(t,'shape',None) is None  and len(t.shape)>0 and t.shape[0]>0:
            X, t_ss, T, E = self.interpolate(t[0])
            ID = self.ID[:X.shape[0]]*0-1

            for n in range(1,t.shape[0]):
                Xn, t_0n, Tn, En = self.interpolate(t[n])

                X = torch.cat((X,Xn),0)
                t_ss = torch.cat((t_ss,t_0n),0)
                T = torch.cat((T,Tn),0)
                E = torch.cat((E,En),0)

                ID = torch.cat((ID, self.ID[:Xn.shape[0]]*0-1),0)

        else:
            X, t_ss, T, E = self.interpolate(t)
            ID = self.ID[:X.shape[0]]*0-1
        
       # return Dataset(X, T-t_0, E)

        return Dataset(torch.cat((t_ss.reshape(-1,1),X.reshape(t_ss.shape[0],-1)),1), T-t_ss, E)
        return TimeVaryingDataset(ID, X, T, E, t_ss)
    


class TimeVaryingDataset(Dataset):
    def __init__(self,ID, X, T, E, t_0, sort=False):
        super().__init__(X,T,E)

         # Copy or convert E
        if torch.is_tensor(ID):
            self.ID = ID.clone()
        else:
            self.ID = torch.tensor(ID).type(torch.int64)
        # Check type
        if not self.ID.dtype == torch.int64:
            warnings.warn("E not bool, continuing by converting it.")
            self.ID = self.ID.type(torch.int64)
        # Check dimensions
        if len(self.ID.shape) > 1:
            warnings.warn("E not onedimensional, continuing by reshaping it.")
            self.ID = self.ID.reshape(-1,)
        # Check size
        if not self.E.shape[0] == self.T.shape[0]:
            raise ValueError("Number of elements in E not equal to shape of first dimension of X")
         # Copy or convert T to onedimensional tensor 
        if torch.is_tensor(t_0):
            self.t_0 = t_0.clone()
        else:
            self.t_0 = torch.tensor(t_0, dtype=torch.get_default_dtype())
        # Check dimensions
        if len(self.t_0.shape) > 1:
            warnings.warn("t_0 not onedimensional, continuing by reshaping it.")
            self.t_0 = self.t_0.reshape(-1,)
        # Check size
        if not self.t_0.shape[0] == self.T.shape[0]:
            raise ValueError("Number of elements in t_0 not equal to shape of first dimension of X")
        

        # Find first index of each ID, assuming each ID comes in a row (no mix)
        self.start_ind =  torch.cat((torch.tensor([0]).to(self.ID),
                        torch.torch.where(self.ID.diff())[0]+1,
                        torch.tensor([self.ID.shape[0]]).to(self.ID)),
                        0)
        self.t_max = self.t_0[self.start_ind[1:]-1]
        
        if sort:
            self.sort()

    def arrange_by(self, indices):
        self.t_0 = self.t_0[indices]
        self.X = self.X[indices]
        self.T = self.T[indices]
        self.E = self.E[indices]
        self.ID = self.ID[indices]    

    def sort(self):
        _, indices = torch.sort(self.t_0, dim=0, stable=True)
        self.arrange_by(indices)
        _, indices = torch.sort(self.ID, dim=0, stable=True)
        self.arrange_by(indices)

    def to_dataset(self):
        # torch.concatenate((self.t_0.reshape(-1,1), self.X),0), self.T.reshape(-1,)-self.t_0.reshape(-1,), self.E

        return Dataset(torch.cat((self.t_0.reshape(-1,1),self.X.reshape(self.t_0.shape[0],-1)),1), 
                       self.T.reshape(-1,)-self.t_0.reshape(-1,), 
                       self.E)

    def split(self, split_proportions):
        if not isinstance(split_proportions, list):
            split_proportions = [split_proportions]
        split_proportions = torch.tensor(split_proportions)

        if split_proportions.sum()>=1:
            split_proportions /= split_proportions.sum()
            split_proportions = split_proportions[:-1]

        unique_IDs = self.ID.unique()
        num_IDs = unique_IDs.shape[0]


        unique_IDs = unique_IDs[ torch.randperm(num_IDs)]

        split_ind = torch.round(split_proportions.cumsum(0)*num_IDs)
        last_ind = 0

        
        datasets = []
        for ind in split_ind:
            cur_IDs = unique_IDs[int(last_ind):int(ind)]
            cur_inds = torch.isin(self.ID,cur_IDs)
            

            datasets.append(TimeVaryingDataset(self.ID[cur_inds],
                                                        self.X[ cur_inds], 
                                                        self.T[cur_inds], 
                                                        self.E[cur_inds],
                                                        self.t_0[cur_inds]))
            last_ind = ind

        cur_IDs = unique_IDs[int(last_ind):]
        cur_inds = torch.isin(self.ID,cur_IDs)
        datasets.append(TimeVaryingDataset(self.ID[cur_inds],
                                                        self.X[ cur_inds], 
                                                        self.T[cur_inds], 
                                                        self.E[cur_inds],
                                                        self.t_0[cur_inds]))
        
        return datasets


    def interpolate(self,t):
        # Index where t_0>0
        I_g = torch.where(self.t_0>t)[0]
        if I_g.shape[0]>0:
            # Index where t_0>0 and first for new ID
            I_u = I_g[torch.where(self.ID[I_g].diff()>0)[0]+1]
            I_u = torch.cat((I_g[0].reshape(1,), I_u), 0)

            I_u = I_u[I_u>0]
            I_u = I_u[ self.ID[I_u] == self.ID[I_u-1]]


            delta = ((self.t_0[I_u]-t)/(self.t_0[I_u]-self.t_0[I_u-1])).reshape(-1,1)

            X_t = self.X[I_u-1]*delta + self.X[I_u]*(1-delta)
            
            # return self.t_0[I_u]*0+t, X_t#, self.T[I_u] - t, self.E[I_u]
            
            if len(X_t.shape) == 1:
                X_t = X_t.reshape(-1,1)

            return X_t, self.t_0[I_u]*0+t, self.T[I_u],  self.E[I_u]
            return torch.cat([(self.t_0[I_u]*0+t).reshape(-1,1), X_t],1), self.T[I_u] - t, self.E[I_u] ,X_t
        
        return  torch.Tensor(0, (self.X.shape[1] if len(self.X.shape)>1  else 1)).to(self.X),torch.Tensor(0,).to(self.t_0), torch.Tensor(0,).to(self.T), torch.Tensor(0,).to(self.E)
        return  torch.Tensor(0, (self.X.shape[1] if len(self.X.shape)>1  else 2)).to(self.X), torch.Tensor(0,).to(self.T), torch.Tensor(0,).to(self.E)


    def resample(self, t):
    
        if not getattr(t,'shape',None) is None  and len(t.shape)>0 and t.shape[0]>0:
            X, t_0, T, E = self.interpolate(t[0])
            ID = self.ID[:X.shape[0]]*0-1

            for n in range(1,t.shape[0]):
                Xn, t_0n, Tn, En = self.interpolate(t[n])

                X = torch.cat((X,Xn),0)
                t_0 = torch.cat((t_0,t_0n),0)
                T = torch.cat((T,Tn),0)
                E = torch.cat((E,En),0)

                ID = torch.cat((ID, self.ID[:Xn.shape[0]]*0-1),0)

        else:
            X, t_0, T, E = self.interpolate(t)
            ID = self.ID[:X.shape[0]]*0-1
        
       # return Dataset(X, T-t_0, E)

        return Dataset(torch.cat((t_0.reshape(-1,1),X.reshape(t_0.shape[0],-1)),1), T-t_0, E)
        return TimeVaryingDataset(ID, X, T, E, t_0)
    



    def __len__(self):
        # print('len ds: ' + str(self.X.shape[0]))
        return self.X.shape[0]

    def __getitem__(self, idx):
        # print(type(idx))

        return torch.concatenate((self.t_0[idx].reshape(1,), self.X[idx].reshape(-1,)),0), self.T[idx]-self.t_0[idx], self.E[idx]
    

class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle=True, batch_size = 1):
        super().__init__(dataset, shuffle=shuffle, batch_size=batch_size)
    

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        self.dataset.regenerate()
        return super()._get_iterator()



class EpochwiseResamplingDataloder(torch.utils.data.DataLoader):
    def __init__(self, dataset, grid_generator, t_m=np.inf, shuffle=True, batch_size = 1, num_workers=0):
        self.grid_generateor = grid_generator
        self.original_dataset=dataset
        self.t_m = t_m
        temp = self.original_dataset.resample(self.grid_generateor())
        temp.right_censor(self.t_m)

        super().__init__(temp, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
    

    def _get_iterator(self) -> '_BaseDataLoaderIter':

        temp = self.original_dataset.resample(self.grid_generateor())
        temp.right_censor(self.t_m)
        self.dataset.X = temp.X
        self.dataset.T = temp.T
        self.dataset.E = temp.E 
        return super()._get_iterator()
    


class MyDataset(TimeVaryingDataset):
    def __init__(self, full_dataset, t_0_max, N_sample=1, t_m = 10):
        self.full_dataset = full_dataset
        self.t_c = np.random.uniform(0, t_0_max,(N_sample,))
        self.t_0_max = t_0_max
        self.N_sample = N_sample

        self.data = self.full_dataset.resample(self.t_c)
        self.data.E[self.data.T>t_m] = False
        self.data.T[self.data.T>t_m] = t_m
        self.t_m = t_m

        self.T = self.data.T


    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __len__(self):
        return self.data.__len__()

    def regenerate(self):
        self.t_c = np.random.uniform(0, self.t_0_max,(self.N_sample,))
        self.data = self.full_dataset.resample(self.t_c)
        self.data.E[self.data.T>self.t_m] = False
        self.data.T[self.data.T>self.t_m] = self.t_m

        self.T = self.data.T


