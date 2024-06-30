from torch.nn import Module
import torch

class MLP(Module):
    def __init__(self, input_size, output_size, hidden_sizes,
                 activation=torch.nn.ReLU, dropout=0.0, batch_norm=False):
        super().__init__()

        self.input_size = input_size

        layer_input_size = input_size
        layerlist = []
        for layer_output_size in hidden_sizes:
            layerlist.append(torch.nn.Linear(
                layer_input_size, layer_output_size))
            layerlist.append(activation())
            if batch_norm:
                layerlist.append(torch.nn.BatchNorm1d(layer_output_size))
            if dropout:
                layerlist.append(torch.nn.Dropout(dropout))

            layer_input_size = layer_output_size

        layerlist.append(torch.nn.Linear(layer_output_size, output_size))

        self.layers = torch.nn.Sequential(*layerlist)

    def forward(self, X):
        logits = self.layers(X)

        return logits