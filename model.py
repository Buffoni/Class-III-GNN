import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

# Define the graph neural network model
class GNN(nn.Module):
    def __init__(self, data):
        super(GNN, self).__init__()
        self.conv1 = gnn.GCNConv(5, 32)
        self.fc1 = nn.Linear(32, 2)
        self.edge_weight1 = nn.Parameter(torch.ones_like(data.edge_attr.clone().detach()), requires_grad=True)

    def forward(self, data):
        x = data.x
        noise = torch.concat(
            (1e-5 * torch.tensor(np.random.randn(2), dtype=torch.float32) * torch.ones((data.x.shape[0], 2)),
             1e-5 * np.random.randn() * torch.ones((data.x.shape[0], 2)), torch.zeros((x.shape[0], 1))),
            dim=1)
        noise.requires_grad = False
        x = x #+ noise
        edge_index = data.edge_index
        edge_weight = F.tanh(self.edge_weight1)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.fc1(x)
        return x
