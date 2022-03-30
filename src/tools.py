import hnswlib
import numpy as np
from disjoint_set import DisjointSet
from tqdm import tqdm_notebook as tqdm
import torch
from scipy.spatial import ConvexHull
from itertools import chain
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def HNSWNearestNeighbors(X, k=2):
    '''For a given set of points X finds k nearest neighbors for every point
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Point data to find nearest neighbors,
        where n_samples is the number of samples
        and n_features is the number of features.

    Returns
    -------
    labels : array-like, shape (n_samples, k)
        indices of k nearest neighbors for every point
    distances : array-like, shape (n_samples, k)
        distances to k nearest neighbors in increasing order
    '''
    X_float32 = np.array(X, dtype=np.float32)
    X_labels = np.arange(len(X))
    
    # Declaring index
    p = hnswlib.Index(space='l2', dim=X.shape[1]) # possible options are l2, cosine or ip
    # Initing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements=len(X), ef_construction=200, M=16)
    # Element insertion (can be called several times):
    p.add_items(X_float32, X_labels)
    # Controlling the recall by setting ef:
    p.set_ef(50 + k) # ef should always be > k
    return p.knn_query(X, k=k)

def GetLowPoints(Cub, f, threshold, num, batch = 100, show_bar = True):
    thetas = np.zeros((0, f.param_dim))
    values = np.zeros(0)
    if show_bar:
        pbar = tqdm(total=num)
    while len(thetas) < num:
        thetas_batch = Cub.get_points(batch)
        values_batch = f(thetas_batch)
        mask = values_batch < threshold
        thetas = np.vstack([thetas, thetas_batch[mask]])
        values = np.concatenate([values, values_batch[mask]])
        if show_bar:
            pbar.update(np.sum(mask))
        del thetas_batch
    return thetas, values

class Cube:
    def __init__(self, min_point, max_point):
        self.min = min_point
        self.max = max_point
        self.dim = max_point.shape[0]
        
    def check_points(self, points):
        return np.logical_and(np.all(self.min < points, axis=1),
                              np.all(self.max > points, axis=1))
    
    def get_points(self, num = 1):
        return self.min + (self.max - self.min)*np.random.rand(num, self.dim)

class NetworkDataSetLoss:
    def __init__(self, hidden_layer_sizes, X, Y, activation='tanh', max_batch_size=500000):
        assert type(X) == torch.Tensor
        assert type(Y) == torch.Tensor
        assert X.device == Y.device
        self.device = X.device
        
        self.max_batch_size = max_batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.b_shapes = tuple(hidden_layer_sizes) + (Y.shape[1],)
        self.W_shapes = tuple(zip(
            [X.shape[1],] + list(hidden_layer_sizes),
            self.b_shapes
        ))
        self.param_dim = np.sum([np.prod(W_shape) for W_shape in self.W_shapes]) + np.sum(self.b_shapes)
        
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        
        self.X = X.clone().detach()
        self.Y = Y.clone().detach()
        
    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.device = device
                
    def predict(self, thetas, with_grads=True):
#         assert thetas.shape[0] <= self.max_batch_size
            
        output = self.X.unsqueeze(0)
        pos = 0
        for d in range(len(self.W_shapes)):

            W_shape, b_shape = self.W_shapes[d], self.b_shapes[d]
            W_len, b_len = np.prod(W_shape), np.prod(b_shape)

            Ws = thetas[:, pos:pos+W_len].reshape(-1, *W_shape)
            pos += W_len

            bs = thetas[:, pos:pos+b_len].reshape(-1, b_shape)
            pos += b_len

#                 output = torch.bmm(output, Ws) + bs[:,None,:]
            output = torch.matmul(output, Ws)
            output.add_(bs.unsqueeze(1))

            if d != len(self.W_shapes) - 1:
                output = self.activation(output)
            del Ws, bs

        output = output if with_grads else output.detach()
#         torch.cuda.empty_cache()
        return output
    
    def __call__(self, thetas, with_grads=True):
        if with_grads:
#             assert len(thetas) <= self.max_batch_size
            Y_pred = self.predict(thetas, with_grads=with_grads)
            losses =  ((Y_pred - self.Y) ** 2).flatten(start_dim=1).mean(dim=1)
            return losses
        
        with torch.no_grad():
            result = []
            for thetas_batch in tqdm(torch.split(thetas, self.max_batch_size)):
                Y_pred = self.predict(thetas_batch, with_grads=with_grads)
                losses =  ((Y_pred - self.Y) ** 2).flatten(start_dim=1).mean(dim=1)
                result.append(losses.detach())
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            return torch.cat(result).detach()
        
class ReLUFixedBiasFullyConnectedNetworkDataSetLoss:
    def __init__(
        self, hidden_layer_sizes, biases, X, Y,
        max_batch_size=500000, lambda_l1=0., lambda_l2=0., last_bias_on=True
    ):
        assert type(X) == torch.Tensor
        assert type(Y) == torch.Tensor
        assert X.device == Y.device
        assert lambda_l1 >= 0
        assert lambda_l2 >= 0
        
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.device = X.device
        self.max_batch_size = max_batch_size

        self.biases = [torch.tensor(bias, dtype=torch.float32, device=self.device) for bias in biases]
        self.last_bias = last_bias_on
            
        self.hidden_layer_sizes = hidden_layer_sizes
        self.b_shapes = tuple(hidden_layer_sizes) + (Y.shape[1],)
        self.W_shapes = tuple(zip(
            [X.shape[1],] + list(hidden_layer_sizes),
            self.b_shapes
        ))
        
        self.param_dim = np.sum([np.prod(W_shape) for W_shape in self.W_shapes]) 
        if last_bias_on:
            self.param_dim += np.prod(self.b_shapes[-1])
        
        self.X = X.clone().detach()
        self.Y = Y.clone().detach()
        
    def to(self, device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.biases = [bias.to(device) for bias in self.biases]
        self.device = device
        
    def predict(self, thetas, inner=False):
        assert (type(thetas) == torch.Tensor) or (type(thetas) == np.ndarray)
        assert len(thetas) <= self.max_batch_size
        
        if type(thetas) == np.ndarray:
            return self.predict(
                torch.tensor(thetas, device=self.device, dtype=torch.float32)
            ).cpu().detach().numpy()
            
        output = self.X.unsqueeze(0)
        pos = 0
        for d in range(len(self.W_shapes)):
            W_shape, b_shape = self.W_shapes[d], self.b_shapes[d]
            W_len, b_len = np.prod(W_shape), np.prod(b_shape)
            Ws = thetas[:, pos:pos+W_len].reshape(-1, *W_shape)
            pos += W_len
            if d < len(self.W_shapes) - 1:
                bs = self.biases[d].reshape(-1, b_shape)
            else:
                if self.last_bias:
                    bs = thetas[:, -b_len:].reshape(-1, b_shape)
                else:
                    bs = torch.zeros((thetas.shape[0], b_shape), device=self.device, dtype=torch.float32)
            output = torch.matmul(output, Ws)
            output.add_(bs.unsqueeze(1))
            
            if d != len(self.W_shapes) - 1:
                output = torch.relu(output)
            if d == len(self.W_shapes) - 1 and not inner:
                ad = ((self.Y - output).flatten(start_dim=1).mean(dim=1))/(1 + self.lambda_l2)
                output.add_(ad.reshape(output.shape[0], 1, 1))
        return output

    def _compute_regularization(self, thetas):
        return self.lambda_l1 * torch.abs(thetas).sum(dim=1) + self.lambda_l2 * ((thetas) ** 2).sum(dim=1)
    
    def __call__(self, thetas):
        assert (type(thetas) == torch.Tensor) or (type(thetas) == np.ndarray)
        
        if type(thetas) == torch.Tensor:
            assert len(thetas) <= self.max_batch_size
#             assert thetas.device == self.device
            Y_pred = self.predict(thetas, inner=True)
            if not self.last_bias:
                addition = ((self.Y - Y_pred).flatten(start_dim=1).mean(dim=1))/(1 + self.lambda_l2)
                Y_pred.add_(addition.reshape(Y_pred.shape[0], 1, 1))
            losses = ((Y_pred - self.Y) ** 2).flatten(start_dim=1).mean(dim=1)
            if not self.last_bias:
                losses.add_(self.lambda_l2*addition**2)
            return losses + self._compute_regularization(thetas)

        with torch.no_grad():
            result = []
            start = 0
            while start < len(thetas):
                thetas_batch = torch.tensor(
                    thetas[start:start+self.max_batch_size],
                    device=self.device, dtype=torch.float32
                )
                result.append(self(thetas_batch).cpu().detach().numpy())
                torch.cuda.empty_cache()
                start += self.max_batch_size
            torch.cuda.empty_cache()
            return np.hstack(result)

            
def make_undirected(graph):
    undirected_graph = graph.tolist()
    for v1 in range(len(graph)):
        for v2 in graph[v1]:
            undirected_graph[v2].append(v1)
    undirected_graph = [list(set(neighbors)) for neighbors in undirected_graph]
        
    return undirected_graph

def make_rectangular(graph):
    max_n = max([len(neighbors) for neighbors in graph])
    for v in range(len(graph)):
        graph[v] += [-1] * (max_n - len(graph[v]))
    return graph


class ExtendedConvexHull(ConvexHull):
    def __init__(self, points, volume_multiplier=1.):
        super().__init__(points, incremental=False)
        self.volume_multiplier = volume_multiplier
        self._initialize_sampler()
        
    def _initialize_sampler(self):
        assert len(self.vertices) > self.ndim
        
        pivot = self.simplices[0][0]
        self._partition_idx = np.array(
            [[pivot] + simplex.tolist() for simplex in self.simplices if pivot not in simplex],
            dtype=int
        )
        partition_vol = np.array([
            np.abs(np.linalg.det(self.points[idx][1:] - self.points[idx][0]))
            for idx in self._partition_idx
        ])
        self._partition_p = partition_vol / np.sum(partition_vol)
        self.mass_center = np.zeros(self.ndim, dtype=float)
        for idx, p in zip(self._partition_idx, self._partition_p):
            self.mass_center += self.points[idx].mean(axis=0) * p
    
    def add_points(self, *args, **kwargs):
        raise Exception('Not supported. Please reinitialize from scratch.')
    
    def sample(self, n_points):
        simplex_idx = np.random.choice(range(len(self._partition_idx)), size=n_points, p=self._partition_p).astype(int)
        points_idx = self._partition_idx[simplex_idx]
        weights = np.random.dirichlet([1.] * (self.ndim+1), size=n_points)
        points = self.points[points_idx.flatten().astype(int)].reshape(*points_idx.shape, -1,)
        points = torch.tensor(points)
        weights = torch.tensor(weights)
        batch = torch.matmul(points.transpose(2, 1), weights[:, :, None]).numpy().sum(axis=2)
        
        # Scaling
        batch = self.mass_center + \
        (batch - self.mass_center) * (self.volume_multiplier ** (1. / batch.shape[1]))
        return batch
    
    def get_dist_to_bounding_planes(self, points):
        inner_dist = (points @ self.equations[:, :-1].T + self.equations[:, -1])
        center_dist = (self.equations[:, :-1] @ self.mass_center + self.equations[:, -1])

        return inner_dist + center_dist
    
def plot_barcodes(result, ax, min_cluster_size=20, title=''):
    minima = result[result.dead_cluster_size >= min_cluster_size].copy()
    minima = minima.sort_values('birth').reset_index()
#     minima.set_index('id_dead_min', inplace=True)
    
    for i, row in minima.iterrows():
        ax.plot([i, i], [row.birth, row.death], c='darkslategrey')
    ax.plot(
        [0, 0],
        [minima.iloc[0].birth, minima.death[minima.death < np.inf].max()],
        c='darkslategrey', linestyle='--'
    )
        
    ax.scatter(
        range(len(minima)),
        minima.birth.values,
        c='mediumblue', marker="v", edgecolor='black',
        zorder=np.inf,
        label='Value at Local Min'
    )
    ax.scatter(
        range(len(minima)), minima.death.values,
        c='red', edgecolor='black', marker='s', zorder=np.inf,
        label='Value at 1-Saddle'
    )

    #ax.set_ylabel('Minima Barcode')
#     ax.set_ylim(-0.03, 0.7)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.legend(fontsize=12)
        
    return ax


def plot_graph(result, ax, min_cluster_size=20, title=''):
    G = nx.DiGraph()
    
    minima = result[result.dead_cluster_size >= min_cluster_size].copy()
    minima = minima.sort_values('birth').reset_index()
    
    for i, row in minima.iterrows():
        G.add_edge(int(row.id_swallowed_min), int(row.id_dead_min), directed=True)
    
    pos=nx.planar_layout(G)
    nx.draw(G, pos, ax=ax, node_size=200, with_labels=False, font_size=10,)
    
def plot_2d_colormap(
    f, ax, 
    scatter_points=None,
    x_min=-1, y_min=-1, x_max=1, y_max=1,
    grid_size=100, nbins=50,
    title=None
):
    X = np.linspace(x_min, x_max, grid_size)
    Y = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.hstack([X[None], Y[None]]).reshape(2, -1).T).reshape(grid_size, grid_size)
    levels = MaxNLocator(nbins=nbins).tick_values(Z.min(), Z.max())
    ax.contourf(X, Y, Z, cmap='coolwarm',levels=levels)
    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$x_{2}$')
    if title:
        ax.set_title(title)
    if scatter_points is not None:
        ax.scatter(
            scatter_points[:, 0], scatter_points[:, 1],
            c='mediumblue', marker="v", edgecolor='black', label='Local Minima'
        )
    ax.legend()
    
def plot_1d(
    f, ax, 
    scatter_points=None,
    x_min=-1, x_max=1,
    grid_size=100,
    title=None
):
    X = np.linspace(x_min, x_max, grid_size).reshape(-1, 1)
    Y = f(X).flatten()
    
    ax.plot(X, Y)
    ax.set_xlabel(r'$x_{1}$')
    ax.set_ylabel(r'$f(x_{1})$')
    if title:
        ax.set_title(title)
    if scatter_points is not None:
        ax.scatter(
            scatter_points[:, 0], f(scatter_points).flatten(),
            c='mediumblue', marker="v", edgecolor='black', label='Local Minima', zorder=np.inf
        )
    ax.legend()
    
# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d', elev=45)

# surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.9)

# thetas_min_np = thetas_min.cpu().detach().numpy()
# ax.scatter(thetas_min_np[:, 0], thetas_min_np[:, 1],
#            f(thetas_min).cpu().detach().numpy(), c='red',zorder=np.inf)

# ax.set_xlabel(r'$x_{1}$')
# ax.set_ylabel(r'$x_{2}$')
# ax.set_zlabel(r'$f(x_{1},x_{2})$')