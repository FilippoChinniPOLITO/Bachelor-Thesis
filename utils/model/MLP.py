from torch import nn


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes

        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))

        self.activation = activation

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

    def get_network_architecture(self):
        return self.layer_sizes
