import numpy as np

class Benchmark:
    def __init__(self, dim=None):
        self.dim = dim
        self.bounds = None

class Alpine01(Benchmark):
    def __init__(self, dim=2):
        super(Alpine01, self).__init__(dim)
        self.bounds = np.array(list(zip([-10.0] * dim, [10.0] * dim)))
        
    def __call__(self, x):
        assert x.shape[1] == self.dim
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)
    

class HumpCamel6(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(HumpCamel6, self).__init__(dim)
        self.bounds = np.array([[-2., 2.], [-1.5, 1.5]])
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        return (4.-2.1* x[:,0] ** 2 + x[:,0] ** 4 / 3) * x[:,0] ** 2 + x[:,0] * x[:,1] + (-4+4* x[:,1] ** 2) * x[:,1] ** 2
    
class Langermann(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(Langermann, self).__init__(dim)
        self.bounds = np.array([[0., 10.], [0., 10.]])
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        a = np.array([3, 5, 2, 1, 7], dtype=float)
        b = np.array([5, 2, 1, 4, 9], dtype=float)
        c = np.array([1, 2, 5, 2, 3], dtype=float)
        
        numerators = np.cos(np.pi * ((np.vstack([x[:,0]] * 5).T - a) ** 2 + (np.vstack([x[:,1]] * 5).T - b) ** 2)) * c
        denominators = np.exp(((np.vstack([x[:,0]] * 5).T - a) ** 2 + (np.vstack([x[:,1]] * 5).T - b) ** 2) / np.pi)

        return -(numerators / denominators).sum(axis=1)
    
class Qing(Benchmark):
    def __init__(self, dim=2):
        super(Qing, self).__init__(dim)
        self.bounds = np.array(list(zip([-self.dim-0.5] * dim, [self.dim+0.5] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        return np.sum((x ** 2 - np.arange(1., self.dim+1, 1)) ** 2, axis=1)

    
class PenHolder(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(PenHolder, self).__init__(dim)
        self.bounds = np.array(list(zip([-11.0] * dim, [11.0] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        a = np.abs(1. - (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi))
        b = np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.exp(a)
        return -np.exp(-np.abs(b) ** -1)
    
    
class Schwefel26(Benchmark):
    def __init__(self, dim=2):
        super(Schwefel26, self).__init__(dim)
        self.bounds = np.array(list(zip([-500.0] * dim, [500.0] * dim)))
        
    def __call__(self, x):
        assert x.shape[1] == self.dim
        return 418.982887 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    
class Tripod(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(Tripod, self).__init__(dim)
        self.bounds = np.array(list(zip([-100.0] * dim, [100.0] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        p1 = (x[:, 0] >= 0).astype(float)
        p2 = (x[:, 1] >= 0).astype(float)
        
        return (p2 * (1.0 + p1) + np.abs(x[:,0] + 50.0 * p2 * (1.0 - 2.0 * p1)) + \
                np.abs(x[:,1] + 50.0 * (1.0 - 2.0 * p2)))
    

class HolderTable(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(HolderTable, self).__init__(dim)
        self.bounds = np.array([(-10., 10.), (-10., 10.)])
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        return -np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(np.abs(1-np.linalg.norm(x, axis=1) / np.pi)))

class UrsemWaves(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(UrsemWaves, self).__init__(dim)
        self.bounds = np.array([(-0.9, 1.2), (-1.2, 1.2)])
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        u = -0.9 * x[:, 0] ** 2
        v = (x[:, 1] ** 2 - 4.5 * x[:, 1] ** 2) * x[:, 0] * x[:, 1]
        w = 4.7 * np.cos(3 * x[:, 0] - x[:, 1] ** 2 * (2 + x[:, 0])) * np.sin(2.5 * np.pi * x[:, 0])
        
        return u + v + w
    
    

class VenterSobiezcczanskiSobieski(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(VenterSobiezcczanskiSobieski, self).__init__(dim)
        self.bounds = np.array(list(zip([-50.0] * dim, [50.0] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        u = x[:, 0] ** 2.0 - 100.0 * np.cos(x[:, 0]) ** 2.0
        v = -100.0 * np.cos(x[:, 0] ** 2.0 / 30.0) + x[:, 1] ** 2.0
        w = - 100.0 * np.cos(x[:, 1]) ** 2.0 - 100.0 * np.cos(x[:, 1] ** 2.0 / 30.0)
        
        return u + v + w
    


class Wavy(Benchmark):
    def __init__(self, dim=2):
        assert dim == 2
        super(Wavy, self).__init__(dim)
        self.bounds = np.array(list(zip([-np.pi] * dim, [np.pi] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim        
        return 1.0 - (1.0 / self.dim) * np.sum(np.cos(10 * x) * np.exp(-x ** 2.0 / 2.0), axis=1)
    
    
class XinSheYang04(Benchmark):
    def __init__(self, dim=2):
        super(XinSheYang04, self).__init__(dim)
        self.bounds = np.array(list(zip([-10.0] * dim, [10.0] * dim)))
    
    def __call__(self, x):
        assert x.shape[1] == self.dim
        
        u = np.sum(np.sin(x) ** 2, axis=1)
        v = np.sum(x ** 2, axis=1)
        w = np.sum(np.sin(np.sqrt(np.abs(x))) ** 2, axis=1)
        return (u - np.exp(-v)) * np.exp(-w)