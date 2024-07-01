from pysare.training._printers import SingleLinePrinter, MultipleLinePrinter
from pysare.training._caching import StateCacher
from torch.nn import Module
import pandas as pd
from numpy.typing import ArrayLike
from time import perf_counter
import string
import os
from pysare.models.SurvivalModel import TorchSurvivalModel
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Union, Iterable, Callable, Literal, TypeVar, NewType, Any
from os import PathLike
import collections
import time
from types import ModuleType
import matplotlib
import torch
import pysare
import numpy as np
import matplotlib.pyplot as plt

from pysare._typing import Batch, AxisLike
# Log
# ==============================================================================
class TrainingLog(pd.DataFrame):
    r"""
    A training log based on pandas DataFrame returned by the PySaRe training routine. 

    Methods
    -------
    plot(backend):
        Plots the training and validation loss.
    """

    def __init__(self, data):
        super(TrainingLog, self).__init__(data)

    def plot(self, backend: Union[Literal['matplotlib', 'plotext'], AxisLike] = 'matplotlib'):
        r"""
        Plots the training and validation loss using the selected backend.

        Parameters
        ----------
            backend (Union[Literal['plotext','matplotlib'], AxisLike]):
                Backend to be used for plotting. If 'matplotlib' (default)or a 
                matplotlib axis, matploblib is used; and if 'plottex', plotex 
                is used. Any other value will be tried to be used as a matplotlib
                axis, for examle will the module plotext work as input.

        Returns
        ----------
            None
        """
        if (backend is None) or (backend == 'matplotlib'):
            _, ax = plt.subplots(1, 1, clear=True)
            backend = 'matplotlib'
        elif backend == 'plotext':
            import plotext
            ax = plotext
            plotext.clear_data()

        elif isinstance(backend, ModuleType):
            ax = backend
            if backend.__name__.startswith('matplotlib'):
                backend = 'matplotlib'
            else:
                backend = ax.__name__
        elif isinstance(backend, matplotlib.axes._axes.Axes):
            ax = backend
            backend = 'matplotlib'
        else:
            ax = backend
            backend = 'unknown'

        ax.plot(self['epoch'], self['training_loss'], label='Training set')

        if hasattr(ax, 'xlabel'):
            ax.xlabel('Epoch')
            ax.ylabel('Loss value')
        else:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss value')

        if 'validation_loss' in self.columns:
            ax.plot(self['epoch'], self['validation_loss'],
                    label='Validation set')

            min_ind = self['validation_loss'].argmin()

            if min_ind and (backend == 'matplotlib'):
                y_min = self['validation_loss'].iloc[min_ind]
                x_min = self['epoch'].iloc[min_ind]
                ax.plot(x_min, y_min, 'og', label=('Validation minimum: \n  epoch '
                                                   + str(int(x_min)) + '\n  loss     ' +
                                                   format(y_min, '.4f')))

                stop_ind = self['epoch'].argmax()
                y_stop = self['validation_loss'].iloc[stop_ind]
                x_stop = self['epoch'].iloc[stop_ind]
                ax.plot(x_stop, y_stop, 'or', label=('Stopped at: \n  epoch '
                                                     + str(int(x_stop)) + '\n  loss     ' +
                                                     format(y_stop, '.4f')))

        if hasattr(ax, 'legend'):
            ax.legend()
        if hasattr(ax, 'show'):
            ax.show()

# Nonfinite gradient functions
# =============================================================================


def nonfinite_gradient_treatment_none(model):
    return True


def nonfinite_gradient_treatment_skip_step(model):
    for _, param in model.named_parameters():
        if (param.requires_grad) and (not torch.isfinite(param.grad).all()):
            # torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            return False
    return True


def nonfinite_gradient_treatment_set_zero(model):
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.grad[~torch.isfinite(param.grad)] = 0.
    return True

# Loss function
# =============================================================================


def negative_log_likelihood(model, batch):
    return -model.log_likelihood(*batch)


def loss_aggregation_mean(loss_vector):
    loss = loss_vector.mean()
    return loss, loss.item()*loss_vector.nelement()


def loss_aggregation_sum(loss_vector):
    loss = loss_vector.sum()
    return loss, loss.item()

# Main class


class Trainer():
    r"""
    Trainer that implements a training routine for PyTorch Modules, in particular  
    PyaRe SurvivalModels.

    Examples and more: 
        https://github.com/oholmer/PySaRe

    Parameters
    ----------
        model (Module):
            The model to be trained. Can be any Pytorch Module that is 
            compatible with the parameters loss_function  and data loaders,
            but typically a SurvivalModel.
        optimizer (Optimizer):
            PyTorch Optimizer used to train the model. 
        lr_scheduler (LRScheduler): 
            PyTorch learning rate scheduler. If lr_schedule_after_batch is
            True, rescheduling is done after each batch (default), otherwize 
            rescheduling is done after each epoch. If None (default), no 
            rescheduling is done.
        lr_schedule_after_batch (bool):
            Determines if learning rate scheduling is done after each epoch 
            (if False, default) or after each batch (if True).
        loss_function (Union[Literal['negative_log_likelihood'],Callable):
            Loss function to be used during training. A function that takes
            the model as first arguement and one of the batches that the 
            data loaders iterates over as second argument, and returns the
            vectorized loss in the form of a tensor. The ouput from this 
            function is made into a scalar based on the loss_aggregation
            parameter. If a the literal 'negative_log_likelihood' (default)
            is passed, negative log-likelihood is used as loss function.
        loss_aggregation (Literal['mean', 'sum']):
            Determines how the loss is aggregated into a scalar. If 'mean' 
            (default) the mean is taken, and if 'sum' the sum is taken.
        nonfinite_gradient_treatment (Literal['none', 'skip_step', 'set_zero']):
            Determinies how nonfinite values (NaN or Inf, 
            based on torch.isfinite) in gradients are treated . If 'none',
            a step is taken without any intervention (often leading to a 
            non-functioning model); if 'skip_step' (default), the step is 
            not taken if any gradient contains a nonfinite element; and if 
            'set_zero' any nonfinite element in a gradient is zet zero. 
            Note that Inf in gradients can be handeled using the 
            pre_step_function or by using hooks, which often helps with 
            NaN as well.
        cache_in_memory (bool):
            Determines if the model is saved in memmory or saved to file.
            If False, the model will be saved to file in the file specified by
            best_model_path if provided and otherwise in a temporary file. 
        best_model_path (PathLike):
            Path where the best model parameters are saved, only used if
            chache_in_memmory is Falase and load_best_model=True is passed 
            when running the train method.
        print_frequency (Float):
            Determines minimal time between printouts as 1/print_frequency.
            If None (default), 
        print_new_lines (bool | int):
            If False (default), progress during trainig will be reprinted 
            on the same linte. If True, each printed epoch is printed on a
            new line. If int, a new line will only be printed if the number of
            epochs since last new line is at least the given value. 

    """

    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 lr_scheduler: Optional[LRScheduler] = None,
                 lr_schedule_after_batch: bool = False,
                 loss_function: Union[Literal['negative_log_likelihood'],
                                      Callable[[Module, Batch], torch.Tensor]] = 'negative_log_likelihood',
                 loss_aggregation: Literal['mean', 'sum'] = 'mean',
                 on_nonfinite_gradient: Literal['none',
                                                       'skip_step', 'set_zero'] = 'none',
                 cache_in_memory: bool = False,
                 best_model_path: Optional[PathLike] = None,
                 print_frequency: Optional[float] = None,
                 print_new_lines: Union[bool, int] = False
                 ) -> None:

        self.model = model
        self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        self.lr_schedule_after_batch = lr_schedule_after_batch

        self.best_model_cache = StateCacher(
            cache_in_memory, best_model_path, overwrite=True)

        if loss_function == 'negative_log_likelihood':
            self.loss_function = negative_log_likelihood
        else:
            self.loss_function = loss_function

        self.loss_aggregation = loss_aggregation
        if loss_aggregation == 'mean':
            self.loss_aggregation_fun = loss_aggregation_mean
        elif loss_aggregation == 'sum':
            self.loss_aggregation_fun = loss_aggregation_sum
        else:
            raise ValueError("loss_aggregation must be 'mean' or 'sum' ")

        # Printer
        # ======================================================================
        if print_frequency is None:
            print_time = 0.0
        else:
            print_time = 1./print_frequency

        if print_new_lines:
            self.printer = MultipleLinePrinter(
                print_time=print_time, print_new_line_count=int(print_new_lines))
        else:
            self.printer = SingleLinePrinter(print_time=print_time)

        if on_nonfinite_gradient == 'none':
            self.on_nonfinite_gradient_fun = nonfinite_gradient_treatment_none
        elif on_nonfinite_gradient == 'skip_step':
            self.on_nonfinite_gradient_fun = nonfinite_gradient_treatment_skip_step
        elif on_nonfinite_gradient == 'set_zero':
            self.on_nonfinite_gradient_fun = nonfinite_gradient_treatment_set_zero
        else:
            raise ValueError(
                "nonfinite_gradient_treatment must be 'none', 'skip_step', or 'set_zero'.")

        self.log = None

    def train(self,
              num_max_epochs: int,
              training_loader: Iterable[Batch],
              validation_loader: Optional[Iterable[Batch]] = None,
              early_stopping_patience: Optional[int] = None,
              load_best_model: bool = True):
        r"""
        Runs the training routine.

        Examples and more: 
            https://github.com/oholmer/PySaRe

        Parameters
        ----------
            num_max_epochs (int): 
                Maximal number of epochs to train.
            training_loader (Iterable[Batch]): 
                Training data loader that iterates over the batches, 
                typically a PyTorch DataLoader.
            validation_loader (Iterable[Batch]):
                Validation data loader that iterates over the batches, 
                typically a PyTorch DataLoader. If None (default), no 
                validation step is used.
            early_stopping_patience (int):
                Number of epochs without improvement before improvement in 
                validation loss before the training is stopped. If None (default),
                early stopping is not used.
            load_best_model (bool):
                Determines if the best model is to be loaded after training.
                True by default.

        Returns
        ----------
            log (TrainingLog):
                A log (pandas DataFrame) of the training and validation loss for
                each epoch. Use log.plot() to visualize the training.
        """
        # Interpret inputs
        # ======================================================================
        if early_stopping_patience is None:
            early_stopping_patience = num_max_epochs

        if self.lr_scheduler:
            if self.lr_schedule_after_batch:
                lr_schedule_batch_step = self.lr_scheduler.step

                def lr_schedule_epoch_step():
                    return None
            else:
                lr_schedule_epoch_step = self.lr_scheduler.step

                def lr_schedule_batch_step():
                    return None
        else:
            def lr_schedule_epoch_step():
                return None
            def lr_schedule_batch_step():
                return None



        # Setup checkpoint path
        if validation_loader is None:
            load_best_model = False
        else:  
            self.best_model_cache.save(self.model)

        # Main training loop
        # ==========================================================================
        train_loss = []
        validation_loss = []

        best_validation_loss = np.inf
        last_epoch_with_improvement = -1

        stopped_early = False
        for epoch in range(1, num_max_epochs+1):

            self.model.train()

            train_epoch_loss = 0.0

            # Loop over train batches
            # ----------------------------------------------------------------------
            for batch_number, batch in enumerate(training_loader):
                # Compute loss
                loss_vector = self.loss_function(self.model, batch)

                train_batch_loss, loss_vector_sum = self.loss_aggregation_fun(
                    loss_vector)

                train_epoch_loss += loss_vector_sum

                # Backpropagation
                self.optimizer.zero_grad()

                train_batch_loss.backward()

                if self.on_nonfinite_gradient_fun(self.model):
                    self.optimizer.step()
                    lr_schedule_batch_step()

                self.printer.train_update(epoch, batch_number+1,
                                          train_batch_loss.item())

            if self.loss_aggregation == 'mean':
                train_epoch_loss = train_epoch_loss / \
                    len(training_loader.dataset)

            train_loss.append(train_epoch_loss)

            lr_schedule_epoch_step()

            # Loop over validation batches
            # ----------------------------------------------------------------------
            if validation_loader:
                self.model.eval()
                validation_epoch_loss = 0.
                with torch.no_grad():
                    for batch in validation_loader:

                        validation_batch_loss_vector = self.loss_function(
                            self.model, batch)
                        _, loss_vector_sum = self.loss_aggregation_fun(
                            validation_batch_loss_vector)
                        validation_epoch_loss += loss_vector_sum

                    if self.loss_aggregation == 'mean':
                        validation_epoch_loss /= len(validation_loader.dataset)

                validation_loss.append(validation_epoch_loss)

                # Early stopping and checkpoint
                # ----------------------------------------------------------------------
                if validation_epoch_loss < best_validation_loss:
                    best_validation_loss = validation_epoch_loss
                    last_epoch_with_improvement = epoch
                    if load_best_model:
                        self.best_model_cache.save(self.model)
                if epoch - last_epoch_with_improvement > early_stopping_patience:
                    stopped_early = True
                    break

            else:
                validation_epoch_loss = None

            self.printer.finalize_update(
                epoch, batch_number+1, train_epoch_loss, validation_epoch_loss, train_batch_loss)

        # ==========================================================================
        # Finalize

        # Test if a checpoints has been saved
        model_restored_from_checkpoint = False
        if load_best_model and (not self.best_model_cache.empty):
            self.best_model_cache.load(self.model)

        self.best_model_cache.delete_temp()

        self.printer.finalize()

        if stopped_early:
            print(
                f"Training stoped at: \n   Epoch: {epoch:.0f}  (no improvement for {early_stopping_patience:.0f} epochs)")
        else:
            print(
                f"Training stoped at:\n   Epoch: {epoch:.0f} (maximal number)")

        if model_restored_from_checkpoint:
            print(
                f"Restored model parameters from:\n   Epoch: {last_epoch_with_improvement:.0f}")

        print(f"   Validation loss: {best_validation_loss:.5f}")

        if validation_loader:
            self.log = TrainingLog(data={'epoch': range(1, epoch+1),
                                         'training_loss': train_loss,
                                         'validation_loss': validation_loss})
        else:
            self.log = TrainingLog(data={'epoch': range(1, epoch+1),
                                         'training_loss': train_loss})
        return self.log

    def plot(self, backend: Union[Literal['matplotlib', 'plotext'], AxisLike] = 'matplotlib'):
        r"""
        Plots the training and validation loss using the selected backend.

        Parameters
        ----------
            backend (Union[Literal['plotext','matplotlib'], AxisLike]):
                Backend to be used for plotting. If 'matplotlib' (default)or a 
                matplotlib axis, matploblib is used; and if 'plottex', plotex 
                is used. Any other value will be tried to be used as a matplotlib
                axis, for example will the module plotext work as input.

        Returns
        ----------
            None
        """
        if self.log is not None:
            return self.log.plot(backend)
        else:
            raise ValueError('Training must be run before plotting log.')