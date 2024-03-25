import torch_geometric.nn as geom_nn
import torch
import torch.nn as nn
from typing import Union
from GNNs import GATConv,GCNConv,APPNP
devices = torch.device('cuda:0')

class model(nn.Module):

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 aggr: str = 'add',
                 **kwargs) -> None:
        super().__init__()

        self.ggnn = geom_nn.GatedGraphConv(out_channels=hidden_channels,
                                           num_layers=num_layers,
                                           aggr=aggr, **kwargs)
        self.gat = GATConv(in_channels=input_channels,
                           out_channels=hidden_channels)
        self.gcn = GCNConv(in_channels=input_channels,
                           out_channels=hidden_channels)
        self.appnp = APPNP(in_channels=input_channels,
                           out_channels=hidden_channels,
                           num_nodes=1600,
                           hidden_dim=116,
                           alpha=0.2,
                           K=10
                           )

        self.gat2 = GATConv(in_channels=hidden_channels,
                            out_channels=hidden_channels
                            )
        self.gcn2 = GCNConv(in_channels=hidden_channels,
                            out_channels=hidden_channels
                            )
        self.appnp2 = APPNP(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            num_nodes=1600,
                            hidden_dim=200,
                            alpha=0.2,
                            K=10
                            )
        self.relu = nn.ReLU
        wide_dimension = hidden_channels + input_channels
        conv_layers_wide = [
            nn.Conv1d(in_channels=200,
                      out_channels=100,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(in_channels=100,
                      out_channels=50,
                      kernel_size=1),
            self.relu(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=50,
                      out_channels=1,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(1, stride=2),
        ]
        self.conv_wide = nn.ModuleList(conv_layers_wide)
        wide_out_dim = wide_dimension
        for layer in conv_layers_wide:
            if not isinstance(layer, self.relu):
                wide_out_dim = self.conv_layer_out_dim(wide_out_dim, layer)
        self.mlp_wide = nn.Sequential(
            nn.Dropout(),
            nn.Linear(wide_out_dim, 1),
        )
        conv_layers_narrow = [
            nn.Conv1d(in_channels=200,
                      out_channels=100,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(3, stride=2),
            nn.Conv1d(in_channels=100,
                      out_channels=50,
                      kernel_size=1),
            self.relu(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(in_channels=50,
                      out_channels=1,
                      kernel_size=3),
            self.relu(),
            nn.MaxPool1d(1, stride=2),
        ]
        self.conv_narrow = nn.ModuleList(conv_layers_narrow)

        narrow_out_dim = hidden_channels
        for layer in conv_layers_narrow:
            if not isinstance(layer, self.relu):
                narrow_out_dim = self.conv_layer_out_dim(narrow_out_dim, layer)
        self.mlp_narrow = nn.Sequential(
            nn.Dropout(),
            nn.Linear(narrow_out_dim, 1),
        )

    def conv_layer_out_dim(self,
                           input_dim: int,
                           layer: Union[nn.Conv1d, nn.MaxPool1d]) -> int:

        layer_params = [layer.kernel_size, layer.padding, layer.stride]
        if isinstance(layer, nn.Conv1d):
            layer_params = [p[0] for p in layer_params]
        kernel, padding, stride = layer_params
        out_dim = (((input_dim + 2 * padding - kernel) / stride) + 1)
        return int(out_dim)

