import random
import pandas as pd
import numpy as np


def permuation(adj, idx1, idx2, num=5):
    """permuation the result

    Args:
        adj (np.array): shape: (num_cells, num_cells)
        idx1 (np.array): shape: (N1,)
        idx2 (np.array): shape: (N2, )
        num (int, optional): iteration numbers. Defaults to 10000.
    """
    ans = list()
    cells = len(adj)


    def swap(cells, idx1, idx2):
        init = list(range(cells))
        for j in list(range(cells))[:: -1]:
            idx = random.randint(0, j)
            tmp = init[j]
            init[j] = init[idx]
            init[idx] = tmp
        l1 = list()
        for j in idx1:
            l1.append(init[j])
        l2 = list()
        for j in idx2:
            l2.append(init[j])
        return l1, l2

    for i in range(num):
        idx1, idx2 = swap(cells, idx1, idx2)
        hit = adj[idx1][:, idx2].sum()
        ans.append(hit)
    return ans

adj = np.random.random((40,40))
adj = np.where(adj>0.5, 1, 0)
idx1 = [1,3,4,5,6,12,23,26]
idx2 = [8,9,13,15,17,18]
ans = permuation(adj, idx1, idx2)
print(ans)