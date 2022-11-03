import networkx as nx
import numpy as np
import scipy.sparse as sprs
from scipy.stats import rankdata
from scipy.spatial import KDTree

import matplotlib.pyplot as pl
from matplotlib.collections import CircleCollection, LineCollection

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

N = 100_000
k = 100

pos, pairs, A = get_RGG(N, k)
N = pos.shape[0]
eigs, vecs = sprs.linalg.eigsh(A,k=1)
amax = eigs[0]
vec = vecs.flatten()
vec /= vec.sum()
print(vec)
#import sys
#sys.exit(1)
eye = sprs.eye(N)
_1 = np.ones(N)

attenuation = alpha = 0.5/amax

#C = sprs.linalg.spsolve(eye-attenuation*A, _1)
#C = C.flatten()
C = vec

cmap = pl.get_cmap('RdPu')

C2 = rankdata(C)
C1 = C.copy()


for C in [C1, C2]:
    C = C.copy()
    C -= C.min()
    C /= C.max()
    fig, ax = pl.subplots(figsize=(10,10))
    R = 4
    collection = CircleCollection(R*np.ones(N), offsets=pos, transOffset=ax.transData,facecolors=cmap(C),edgecolor='#666666',linewidths=0.1)
    ax.add_collection(collection)


    segments = [ (pos[i,:], pos[j,:]) for i, j in pairs ]
    collection = LineCollection(segments,color='#bbbbbb',zorder=-1,lw=1)
    #ax.add_collection(collection)

    fig.patch.set_facecolor('#222222')
    pl.rcParams['axes.facecolor'] = '#222222'

    pl.axis('square')
    pl.axis('off')


pl.show()



