import torch
import warnings
import numpy as np


def default_input_function(model, X, T, E=None):
    """ This functions creates inputs to the neural network so that the array 
    outputs from log_likelihood, survival_function,  and lifetime_density and 
    can be calculated.

    Let Y be the desired array. The function calculates inputs to give a vectorized 
    version of y (=Y.reshape(-1,)) and an output function that reshapes this 
    matrix to Y. 

    It also tries to find the correct device and dtype of the network.
    """

    # Finde device and dtype of model
    parameters = model.parameters()
    if next(parameters, None) is None:
        # Model parameters empty
        warnings.warn(
            "Could not determine device and dtype from self.parameters(), using 'cpu' and torch.float")
        device = "cpu"
        dtype = torch.float
    else:
        first_parameter = next(parameters)
        device = first_parameter.device
        dtype = first_parameter.dtype

    X = torch.Tensor(X).to(device=device, dtype=dtype)
    N = X.shape[0]
    T = torch.tensor(T).to(device=device, dtype=dtype)
    if not E is None:
        E = torch.Tensor(E).to(device=device, dtype=dtype)
        if not E.shape == T.shape:
            raise ValueError("Dimensions of T and E not equal.")

    # If T is onedimensional
    transpose_output = len(T.shape) == 1
    if transpose_output:
        T = T.reshape(1, -1)
        if not E is None:
            E = E.reshape(1, -1)

    if len(T.shape) > 2:
        raise ValueError("Number of dimensions of T larger than 2")
    elif (len(T.shape) == 0) or (T.shape[0] == 1):
        # T is a row vector with times to apply to each row in X
        T = T.reshape(1, -1)
        T = T.repeat_interleave(N, 0).reshape(-1,)
        M = T.shape[0]//N
        X = X.repeat_interleave(M, 0)

        if not E is None:
            E = E.reshape(1, -1)
            E = E.repeat_interleave(N, 0).reshape(-1,)

    elif T.shape[0] == N:
        # T is a column vector or matrix for witch row i is applied to row i in X
        T = T.reshape(-1,)
        M = T.shape[0]//N
        X = X.repeat_interleave(M, 0)

        if not E is None:
            E = E.reshape(-1,)

    else:
        raise ValueError("Dimension of T not compatible with X")

    def output_function(output):
        if transpose_output:
            return np.array(output.reshape(N, M).detach()).T

        return np.array(output.reshape(N, M).detach())

    if E is None:
        return X, T, output_function

    else:
        return X, T, E, output_function
