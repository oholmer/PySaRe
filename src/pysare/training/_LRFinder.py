import torch
import matplotlib.pyplot as plt
import torch_lr_finder


def negative_log_likelihood(model, batch):
    return -model.log_likelihood(*batch)


class WrapLoss(torch.nn.Module):
    r"""A Wrapper used to wrap the loss for lr_finder
    """

    def __init__(self, model, loss_function=negative_log_likelihood, loss_aggregation='mean') -> None:
        super().__init__()

        self.model = model
        self.loss_function = loss_function

        if loss_aggregation == 'mean':
            self.loss_aggregation = torch.mean
        elif loss_aggregation == 'sum':
            self.loss_aggregation = torch.sum
        else:
            raise ValueError("loss_aggregation must be 'mean' or 'sum'")

    def forward(self, batch: torch.Tensor, _: torch.Tensor) -> torch.Tensor:

        model = self.model
        loss = self.loss_function(self.model, batch)

        return self.loss_aggregation(loss)


class WrapModel(torch.nn.Module):
    r"""A Wrapper used to wrap a model for lr_finder
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch):
        return batch


class WrapDataLoaders(object):
    r"""A Wrapper used to change the collate function of a datalaoder
    in lr_finder"""

    def __init__(self, train_loader, val_loader=None):

        self.original_train_collate_fn = train_loader.collate_fn
        self.train_loader = train_loader

        self.val_loader = val_loader
        if val_loader:
            self.original_val_collate_fn = val_loader.collate_fn

    def __enter__(self):

        self.original_train_collate_fn = self.train_loader.collate_fn

        def train_collate_fn(batch):
            return self.original_train_collate_fn(batch), None

        self.train_loader.collate_fn = train_collate_fn

        if self.val_loader:
            def val_collate_fn(batch):
                return self.original_train_collate_fn(batch), None

            self.original_val_collate_fn = self.val_loader.collate_fn
            self.val_loader.collate_fn = val_collate_fn

        return self.train_loader, self.val_loader

    def __exit__(self, *args):
        self.train_loader.collate_fn = self.original_train_collate_fn
        if self.val_loader:
            self.val_loader.collate_fn = self.original_val_collate_fn


class LRFinder:
    r"""Learning rate range test based on the package torch_lr_finder. 

    The learning rate range test increases the learning rate in a pre-training 
    run between two boundaries in a linear or exponential manner. The result 
    can be used to determine an appropriate learning rate, see references below.

    Arguments:
    ----------

        model (Module): 
            Model.
        optimizer (torch.optim.Optimizer): 
            Optimizer where the defined learning is assumed to be the lower 
            boundary of the range test.
        loss_function (torch.nn.Module): 
            wrapped loss function.
        memory_cache (bool): 
            If this flag is set to True, `state_dict` of model and optimizer 
            will be cached in memory. Otherwise, they will be saved to files 
            under the `cache_dir`.
        cache_dir (string): 
            path for storing temporary files. If no path is specified, 
            system-wide temporary directory is used (default). Notice that this
            parameter will be ignored if `memory_cache` is True.

    Examples:
    ---------

        >>> lr_finder = LRFinder(model, optimizer)
        >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100, reset=False)
        >>> lr_finder.plot() # to inspect the loss-learning rate graph
        >>> lr_finder.reset() # to reset the model and optimizer to their initial 
                              # state, done automatically if reset=True above

    
    References:
    -----------

        Cyclical Learning Rates for Training Neural Networks: 
            https://arxiv.org/abs/1506.01186
        fastai/lr_find: 
            https://github.com/fastai/fastai
    """

    def __init__(self, model, optimizer, loss_function=negative_log_likelihood,
                 loss_aggregation='mean', memory_cache=True, cache_dir=None) -> None:

        self.backend_lr_finder = torch_lr_finder.LRFinder(model=WrapModel(model),
                                                          optimizer=optimizer,
                                                          criterion=WrapLoss(
                                                              model, loss_function, loss_aggregation),
                                                          memory_cache=memory_cache,
                                                          cache_dir=cache_dir)
        self.suggested_lr = None

    def range_test(
        self,
        train_loader,
        val_loader=None,
        start_lr=None,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
        accumulation_steps=1,
        non_blocking_transfer=True,
        reset=True
    ):
        r"""Performs the learning rate range test. 

        Arguments:
        -------
            train_loader (Iterable[Batch]):
                The training set data loader. The dataloader should iterate 
                over batches that are used as inputs to loss_function.
            val_loader (Iterable[Batch]): 
                When given a data loader, the model is evaluated after each 
                iteration on that dataset and the evaluation loss is used. Note 
                that in this mode the test takes significantly longer, but 
                generally produces more precise results. Default: None.
            start_lr (float): 
                The starting learning rate for the range test.
                Default: None (uses the learning rate from the optimizer).
            end_lr (float): 
                The maximum learning rate to test. Default: 10.
            num_iter (int): 
                The number of iterations over which the test occurs. 
                Default: 100.
            step_mode (str): 
                One of the available learning rate policies, linear or 
                exponential ("linear", "exp"). Default: "exp".
            smooth_f (float): 
                The loss smoothing factor within the [0, 1[ interval. 
                Disabled if set to 0, otherwise the loss is smoothed using
                exponential smoothing. Default: 0.05.
            diverge_th (int): 
                The test is stopped when the loss surpasses the threshold:  
                diverge_th * best_loss. Default: 5.
            accumulation_steps (int): 
                steps for gradient accumulation. If it is 1, gradients are not 
                accumulated. Default: 1.
            non_blocking_transfer (bool): 
                when non_blocking_transfer is set, tries to convert/move data 
                to the device asynchronously if possible, e.g., moving CPU 
                Tensors with pinned memory to CUDA devices. Default: True.
            reset (bool): 
                If True, the the model and optimizer is restored to  their 
                initial states. Default: True.

        Examples:
        -------
        Example (fastai approach):
            >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
            >>> lr_finder.range_test(dataloader, end_lr=100, num_iter=100)

        Example (Leslie Smith's approach):
            >>> lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
            >>> lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="linear")

        Gradient accumulation is supported; example:
            >>> train_data = ...    # prepared dataset
            >>> desired_bs, real_bs = 32, 4         # batch size
            >>> accumulation_steps = desired_bs // real_bs     # required steps for accumulation
            >>> dataloader = torch.utils.data.DataLoader(train_data, batch_size=real_bs, shuffle=True)
            >>> acc_lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
            >>> acc_lr_finder.range_test(dataloader, end_lr=10, num_iter=100, accumulation_steps=accumulation_steps)

        Reference:
        [Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](
        https://medium.com/huggingface/ec88c3e51255)
        [thomwolf/gradient_accumulation](https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3)
        """
        with WrapDataLoaders(train_loader, val_loader) as (wraped_train_loader, wraped_val_loader):
            # range_test currently has no return, but for possible future
            # compatibility it is assumed it might have
            result = self.backend_lr_finder.range_test(
                train_loader,
                val_loader,
                start_lr,
                end_lr,
                num_iter,
                step_mode,
                smooth_f,
                diverge_th,
                accumulation_steps,
                non_blocking_transfer)
        if reset:
            self.reset()
        return result

    def plot(self,
             skip_start=10,
             skip_end=5,
             log_lr=True,
             show_lr=None,
             ax=None,
             suggest_lr=True):
        r"""Plots the learning rate range test.

        Arguments:
        ----------
            skip_start (int): number of batches to trim from the start.
                Default: 10.
            skip_end (int): number of batches to trim from the start.
                Default: 5.
            log_lr (bool): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. Default: True.
            show_lr (float): if set, adds a vertical line to visualize the
                specified learning rate. Default: None.
            ax (matplotlib.axes.Axes): the plot is created in the specified
                matplotlib axes object and the figure is not be shown. If `None`, 
                then the figure and axes object are created in this method and 
                the figure is shown . Default: None.
            suggest_lr (bool): 
                If True, suggest a learning rate by 'steepest': the point with 
                steepest gradient (minimal gradient). You can use that point as 
                a first guess for an LR. Default: True.

        Returns:
        --------
            The matplotlib.axes.Axes object that contains the plot,
            and the suggested learning rate (if set suggest_lr=True).
        """
        result = self.backend_lr_finder.plot(skip_start,
                                             skip_end,
                                             log_lr,
                                             show_lr,
                                             ax,
                                             suggest_lr)
        if suggest_lr:
            self.suggested_lr = result[1]

        return result

    def set_lr(self, lr='suggested'):
        r"""Sets the learning rate of the optimizer

        Arguments:
        ----------
            lr ('suggested', float): 
            learning rate to be set. If 'suggested', the suggested learning 
            rate is used, the method plot must have been run with argument 
            suggest_lr True before this. If float, this value is set.
        """

        if lr == 'suggested':
            if self.suggested_lr:
                lr = self.suggested_lr
            else:
                raise ValueError(('Suggested learning rate not calculated.',
                                  'run method plot  with argument suggest_lr True.'))

        for group in self.backend_lr_finder.optimizer.param_groups:
            group['lr'] = lr

    def reset(self):
        r"""Restores the model and optimizer to their initial states. Is run 
        by default in range_test if parameter reset i not set to false."""
        return self.backend_lr_finder.reset()
