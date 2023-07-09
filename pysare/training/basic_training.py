import torch
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

def loss_fucntion(model, X, T, E):
    return -model._log_likelihood(X, T, E).sum()

def basic_training(model, training_loader, optimizer, num_epochs, device="cpu", dtype=torch.float32,
            print_progress=True, validation_loader = None, tb_log_dir = None, best_model_checkpoint_path=None):


        best_loss = np.inf

        if tb_log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(tb_log_dir)

        log = BasicLog(data = {'epoch': np.zeros(num_epochs,),
                                   'training_loss': np.zeros(num_epochs,),
                                   'test_loss': np.zeros(num_epochs, )})

        for t in range(num_epochs):
            update_fun = getattr(training_loader.dataset, "update", None)
            if callable(update_fun):
                training_loader.dataset.update()
                training_loader = torch.utils.data.DataLoader(training_loader.dataset, 
                                              shuffle=True, 
                                              batch_size = training_loader.batch_size)

            if print_progress:
                print(f"Epoch {t + 1}\n-------------------------------")
            train_loss, test_loss = train_epoch(model, training_loader, optimizer,
                                                      device,dtype, print_progress,
                                                      validation_loader)
            log.iloc[t]['epoch'] = t+1
            log.iloc[t]['training_loss'] = train_loss
            log.iloc[t]['test_loss'] = test_loss

            if tb_log_dir is not None:
                writer.add_scalar('Training loss', train_loss, t)
                writer.add_scalar('Test loss', test_loss, t)


            if best_model_checkpoint_path is not None:
                if test_loss < best_loss:
                    best_loss=test_loss
                    torch.save({
                    'model_state_dict': model.state_dict(),
                    }, best_model_checkpoint_path)

        if best_model_checkpoint_path is not None:
    
            checkpoint = torch.load(best_model_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        print("Done!")
        return log

def train_epoch(model, dataloader, optimizer, device="cpu", dtype=torch.float32,
                     print_progress=True, test_dataloader = None):

        size = len(dataloader.dataset)
        model.train()

        train_loss = 0.
        for batch, (X, T, E) in enumerate(dataloader):
            X, T, E = X.to(device=device, dtype=dtype), T.to(device=device, dtype=dtype), E.to(device)

            # Compute prediction error
            # pred = model(X)
            # loss = loss_fn(pred, y)
            loss = loss_fucntion(model, X, T, E)
            train_loss += loss.item()
            # loss /= X.shape[0]

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            take_step = True
            for name, param in model.named_parameters():
                if (param.requires_grad) and (not torch.isfinite(param.grad).all()):
                    #torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    take_step = False
                    break

            if take_step:
                optimizer.step()

            loss /= X.shape[0]
            
            if print_progress and batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss = train_loss/dataloader.dataset.T.shape[0]

        if test_dataloader is not None:

            model.eval()
            test_loss = 0.
            with torch.no_grad():
                for X, T, E in test_dataloader:
                    X, T, E = X.to(device=device, dtype=dtype), T.to(device=device, dtype=dtype), E.to(device=device)
                    test_loss += loss_fucntion(model, X, T, E).item()

            test_loss /= test_dataloader.dataset.T.shape[0]

            if print_progress:
                print(f"Test loss:     {test_loss:>7f} ")

            return train_loss, test_loss
        else:
            return train_loss, np.nan

class BasicLog(pd.DataFrame):
    def __init__(self,data):
        super(BasicLog, self).__init__(data)

    def plot(self, fig_num=None, ax=None):
        
        if fig_num:
            fig, ax = plt.subplots(1, 1, num=fig_num, clear=True)
        elif not ax:
            fig, ax = plt.subplots(1, 1, clear=True)
        

        
        ax.plot(self['epoch'], self['training_loss'])
        ax.plot(self['epoch'], self['test_loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss value')

        min_ind = self['test_loss'].argmin()
        if min_ind:
            y_min = self['test_loss'].loc[min_ind]
            x_min = self['epoch'].loc[min_ind]
            ax.plot(x_min, y_min, 'og')
        #ax.legend(['Training set', 'Test set', 'Test minimum: \n  epoch '
        #           + str(int(x_min))+ '\n  loss     '+format(y_min, '.4f')])


        stop_ind = self['epoch'].argmax()
        y_stop = self['test_loss'].loc[stop_ind]
        x_stop = self['epoch'].loc[stop_ind]
        ax.plot(x_stop, y_stop, 'or')

        if min_ind:
            ax.legend(['Training set', 'Test set', 'Test minimum: \n  epoch '
                    + str(int(x_min))+ '\n  loss     '+format(y_min, '.4f'),
                    'Stopped at: \n  epoch '
                    + str(int(x_stop)) + '\n  loss     ' + format(y_stop, '.4f')
                    ])
        else:
            ax.legend(['Training set', 'Test set'])