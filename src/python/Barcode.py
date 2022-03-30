import numpy as np
import pandas as pd


def _UnionFatherSearch(idx, union_array):
    if union_array[idx] == idx:
        return idx
    union_array[idx] = _UnionFatherSearch(union_array[idx], union_array)
    return union_array[idx]


class BarcodeSearch():
    '''
    BarcodeSearch: Class for calculating zero sublevel homology of a point cloud.
    '''
    def __init__(self, values, graph, dist=None):
        '''
        Parameters
        ----------
                
        values : np.array of shape N, where N is size of point cloud.
                 Values of function on point of cloud
        
        graph : np.array of shape N * K. 
                graph[i] contents indexes of points closest 
                to point with index i. For example this array
                can be calculated by some ANN algorithm. 
                
        dist : np.array of shape N * K. graph[i] is distances
               between points with index i and points with index graph[i]
               By default these distances 
        

        '''
        self.table = graph
        self.values = values
        if dist is None:
            dist = np.ones_like(graph)
        self.dist = dist


    def Initialize(self):
        '''
        Initial sort of point cloud
        '''
        self.order = np.argsort(self.values)
        self.back_order = np.empty(self.order.size, self.order.dtype)
        self.back_order[self.order] = np.arange(self.back_order.size)
        
        self.graph = [[] for _ in range(len(self.values))]
        for vert in range(self.table.shape[0]):
            for neu_idx in range(self.table.shape[1]):
                neu = self.table[vert][neu_idx]
                if neu == vert or neu == -1:
                    continue
                if self.back_order[vert] < self.back_order[neu]:
                    self.graph[neu].append((self.dist[vert, neu_idx], vert))
                else:
                    self.graph[vert].append((self.dist[vert, neu_idx], neu))
        for vertex in self.graph:
            vertex.sort()
        

    def ComputeBarcode(self, rad = None, FindWays=False):
        '''Casculate zero sublevel homology of point up to radius.

        Parameters
        ----------
        rad : maximum radius in up to which we search for neighbors.
              If rad is None, it will be 2*(mean of initial distances)
        FindWays : bool, False by default.
        
        Return
        ------
        pandas Dataframe with columns: ['birth', 'death', 'birth_swallowing_cluster',
                'id_dead_min', 'id_saddle', 'id_swallowing_min', 
                'dead_cluster_size', 'swallowing_cluster_size', 'mean_height_cluster']
        '''
        if rad is None:
            rad = 2*self.dist.mean()
        father = -np.ones_like(self.order)
        answer = dict()
        if FindWays:
            tree = [[] for _ in range(len(self.values))]
        
        for vertex in self.order:
            near_clusters = dict()
            for dist, neu_idx in self.graph[vertex]:
                if dist > rad:
                    break
                neu_father = _UnionFatherSearch(neu_idx, father)
                prev_neu = near_clusters.get(neu_father)
                if prev_neu is None or self.back_order[neu_idx] < self.back_order[prev_neu]:
                    near_clusters[neu_father] = neu_idx
            
            if len(near_clusters) == 0:
                
                # In this case we create new cluster:
                #    0 - 'birth',  1 - 'death', 2 - 'birth_swallowing_cluster',
                #    3 - 'id_dead_min', 4 - 'id_saddle', 5 - 'id_swallowing_min', 
                #    6 - 'dead_cluster_size', 7 - 'swallowing_cluster_size', 8 - 'mean_height_cluster'
                
                answer[vertex] = [self.values[vertex], np.inf, np.nan, vertex, 
                                  np.nan, np.nan, 1, np.nan, 0.0]
                father[vertex] = vertex
            elif len(near_clusters) == 1:
                # In this case we just add point to cluster
                
                cluster = next(iter(near_clusters.keys()))
                answer[cluster][6] += 1
                answer[cluster][8] += self.values[vertex] - self.values[cluster]
                father[vertex] = cluster
                if FindWays:
                    tree[vertex].append(near_clusters[cluster])
                    tree[near_clusters[cluster]].append(vertex)
            else:
                #In this case we join all near clusters
                
                glob = self.order[min(self.back_order[x] for x in near_clusters.keys())]
                father[vertex] = glob
                answer[glob][6] += 1
                glob_size = answer[glob][6]
                for cluster in near_clusters.keys():
                    if cluster != glob:
                        father[cluster] = glob
                        answer[cluster][1] = self.values[vertex]
                        answer[cluster][2] = self.values[glob]
                        answer[cluster][4] = vertex
                        answer[cluster][5] = glob
                        answer[cluster][7] = glob_size
                        answer[glob][6] += answer[cluster][6]
                        answer[glob][8] += answer[cluster][8] + (self.values[cluster] - 
                                                                 self.values[glob]) * answer[cluster][6]
                    if FindWays:
                        tree[vertex].append(near_clusters[cluster])
                        tree[near_clusters[cluster]].append(vertex)
        
        answer_arr = np.array(list(answer.values()), dtype=object)
        answer_arr[:,8] /=  answer_arr[:,6]
                        
        table = pd.DataFrame(answer_arr, 
                             columns=['birth', 'death', 'birth_swallowing_cluster',
                                      'id_dead_min', 'id_saddle', 'id_swallowing_min', 'dead_cluster_size',
                                      'swallowing_cluster_size', 'mean_height_cluster'], dtype=object)
        if FindWays:
            searcher = PathSearcher(tree)
            searcher.Initialize()
            return table, searcher 
            
        return table

class PathSearcher():
    def __init__(self, tree_list):
        self.tree_list = tree_list
    
    def Initialize(self):
        valency_list = np.array([len(neu_list) for neu_list in self.tree_list])
        visits = -np.ones_like(valency_list)
        self.degree = -np.ones_like(valency_list)
        self.tree = -np.ones_like(valency_list)
        current = 0
        self.degree[current] = 0
        visits[0] = 0
        seen = 1
        while seen < len(self.degree):
            if visits[current] == valency_list[current]:
                current = self.tree[current]
            else:
                next_vertex = self.tree_list[current][visits[current]]
                visits[current] += 1
                if visits[next_vertex] < 0:
                    self.tree[next_vertex] = current
                    self.degree[next_vertex] = self.degree[current] + 1
                    visits[next_vertex] = 0
                    seen += 1
                    current = next_vertex

    def GetWay(self, start, finish):
        start_way = [start]
        finish_way = []
        while start != finish:
            deg_start, deg_finish = self.degree[start], self.degree[finish]
            if deg_start <= deg_finish:
                finish_way.append(finish)
                finish = self.tree[finish]
            if deg_start >= deg_finish:
                start = self.tree[start]
                start_way.append(start)
        finish_way.reverse()
        return start_way+finish_way
