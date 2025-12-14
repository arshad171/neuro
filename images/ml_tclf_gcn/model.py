import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import torch as th
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot
from torch_geometric.utils import remove_self_loops, add_self_loops
import numpy as np
import random
from torch_geometric.data import Data
import numpy as np

batch_size = 4
num_feats = 784
hidden_dim = 64
num_classes = 20
alpha = 0.2
beta = 0.05
lam = 0.5


class GCNIIdenseConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 **kwargs):

        super(GCNIIdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = False

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, alpha, h0, beta, edge_weight=None):
        """"""

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        support = (1 - beta) * (1 - alpha) * x + (1 - alpha) * beta * torch.matmul(x, self.weight1)
        initial = (1 - beta) * (alpha) * h0 + beta * (alpha) * torch.matmul(h0, self.weight2)

        out = self.propagate(edge_index, x=support, norm=norm) + initial
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class batch_norm(torch.nn.Module):
    def __init__(self, dim_hidden, type_norm, skip_connect=False, num_groups=1,
                 skip_weight=0.005):
        super(batch_norm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden


    def forward(self, x):
        if self.type_norm == 'None':
            return x

        else:
            raise Exception(f'the normalization has not been implemented')

class GCNII_model(torch.nn.Module):
    def __init__(self):
        super(GCNII_model, self).__init__()

        self.layers = th.nn.ModuleList([
            torch.nn.Linear(num_feats, hidden_dim),
            GCNIIdenseConv(hidden_dim, hidden_dim),
            batch_norm(hidden_dim, "None", False, 0.005),
            GCNIIdenseConv(hidden_dim, hidden_dim),
            batch_norm(hidden_dim, "None", False, 0.005),
            torch.nn.Linear(hidden_dim, num_classes),
        ])
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(x)
        hx = self.layers[0](x)

        x = self.layers[1](hx, edge_index, alpha, hx, beta, edge_weight)
        x = self.layers[2](x)

        x = self.layers[3](x, edge_index, alpha, hx, beta, edge_weight)
        x = self.layers[4](x)

        x = F.relu(x)

        x = self.layers[5](x)

        return F.log_softmax(x, dim=1)

class Model:
    def __init__(self, device_type="cpu", batch_size=batch_size):
        self.batch_size = batch_size
        print("cuda available:", th.cuda.is_available())
        self.device = th.device(device_type)

        self.model = GCNII_model().to(self.device)

    def get_data(self, batch_size=batch_size):
        # x = np.loadtxt("./ml_models_tc/merged.txt")
        # x = torch.tensor(x, dtype=torch.float32)
        # y = np.loadtxt("./ml_models_tc/label.txt")
        # y = torch.tensor(y, dtype=torch.long)


        l = 1_000
        x = torch.tensor(np.random.rand(l, 784), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.random.randint(0, 1, size=(l,)), dtype=torch.long).to(self.device)

        edges = []
        for _ in range(20 * l):
            edges.append([random.randint(0, l-1), random.randint(0, l-1)])

        # edges = []

        # with open("./ml_models_tc/graph.txt", 'r') as file:
        #     for line in file:
        #         parts = line.split()
        #         if len(parts) == 2:
        #             edges.append([int(parts[0]), int(parts[1])])

        edges = np.array(edges).T
        edges = th.tensor(edges, device=self.device)

        data = Data(x=x, y=y, edge_index=edges)
        # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
        num_nodes = data.x.size(0)
        if not isinstance(data.edge_index, torch.Tensor):
            data.edge_index = torch.from_numpy(data.edge_index).long()

        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index

        # train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # train_num = 10
        # val_num = 5
        # for i in range(20):  # number of labels
        #     index = (data.y == i).nonzero()[:, 0]
        #     perm = torch.randperm(index.size(0))
        #     train_mask[index[perm[:train_num]]] = 1
        #     val_mask[index[perm[train_num:(train_num + val_num)]]] = 1
        #     test_mask[index[perm[(train_num + val_num):]]] = 1
        # data.train_mask = train_mask
        # data.val_mask = val_mask
        # data.test_mask = test_mask

        data = data.to(self.device)
        return data

    def predict(self, batch_size=batch_size):
        batch_size = self.batch_size if not batch_size else batch_size
        data = self.get_data(batch_size)

        with th.no_grad():
            preds = self.model(data)

            # return preds.argmax(dim=1).detach().numpy().tolist()
            return preds.cpu().argmax(dim=1).detach().numpy().tolist()

# def predict(batch_size=4):

if __name__ == "__main__":
    import time
    import argparse
    import json
    import os

    import setproctitle
    setproctitle.setproctitle("my_proc")

    time_cap = 1 * 30

    # time.sleep(10)

    parser = argparse.ArgumentParser(description="kwargs")

    parser.add_argument("--batch-size", type=int, help="batch size", default=1)
    parser.add_argument("--out-folder", type=str, help="output folder", default=".")

    args = parser.parse_args()
    arg_batch_size = args.batch_size
    arg_out_folder = args.out_folder

    # json.dump({"pid": os.getpid()}, open(os.path.join(arg_out_folder, "pid.json"), "w"))

    batch_size = arg_batch_size

    model = Model(device_type="cuda", batch_size=arg_batch_size)

    sts = []
    # for ix in range(1000):
    start_time = time.time()
    while time.time() - start_time < time_cap:
    # for _ in range(100):
        t1 = time.time()
        model.predict()
        print(time.time() - t1, flush=True)
        sts.append(time.time() - t1)
        # time.sleep(0.1)

    while True:
        print("alive", flush=True)
        time.sleep(1)
    
    # json.dump({"sts": sts}, open(os.path.join(arg_out_folder, "sts.json"), "w"))
