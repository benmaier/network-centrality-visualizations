import networkx as nx
import numpy as np
import scipy.sparse as sprs
from scipy.stats import rankdata

from scipy.spatial import KDTree

def get_RGG(N, k):
    neighs_within_r = k
    scale_N = 4/np.pi
    bigN = scale_N * N
    density = bigN/4
    r = np.sqrt(neighs_within_r/density/np.pi)
    pos = np.random.rand(int(bigN),2)*2-1
    ndx = np.where(np.sqrt((pos**2).sum(axis=1))<1)[0]
    pos = pos[ndx,:]
    N = pos.shape[0]
    kd_tree = KDTree(pos)
    pairs = kd_tree.query_pairs(r=r)
    A = sprs.lil_matrix((N, N))
    for i, j in pairs:
        A[i,j] = 1.0
        A[j,i] = 1.0
    return pos, pairs, A.tocsr()




#print(get_RGG(N, k))

#p = k/(N-1)
#G = nx.fast_gnp_random_graph(N,p)

N = 30_000
k = 30

#A = nx.to_scipy_sparse_array(G,dtype=float)
pos, pairs, A = get_RGG(N, k)
N = pos.shape[0]
_A  = A.toarray()
#print(type(_A))
amax = sprs.linalg.eigsh(A,k=1,return_eigenvectors=False)[0]
eye = sprs.eye(N)
_eye = eye.toarray()
_1 = np.ones(N)

attenuation = alpha = 0.5/amax

#C = (sprs.linalg.inv(eye-attenuation*A)-eye).dot(_1)
C = (np.linalg.inv(_eye-attenuation*_A)-_eye).dot(_1)
C = C.flatten()

import matplotlib.pyplot as pl
from matplotlib.collections import CircleCollection, LineCollection


cmap = pl.get_cmap('RdPu')

C2 = rankdata(C)
C1 = C.copy()
print(C2)


for C in [C1, C2]:
    C = C.copy()
    C -= C.min()
    C /= C.max()
    print(C)
    fig, ax = pl.subplots(figsize=(10,10))
    R = 7
    collection = CircleCollection(R*np.ones(N), offsets=pos, transOffset=ax.transData,facecolors=cmap(C),edgecolor='#666666',linewidths=0.5)
    ax.add_collection(collection)


    segments = [ (pos[i,:], pos[j,:]) for i, j in pairs ]
    collection = LineCollection(segments,color='#bbbbbb',zorder=-1,lw=1)
    ax.add_collection(collection)
    pl.axis('square')
    pl.axis('off')

pl.show()



