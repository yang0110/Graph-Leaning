import numpy as np
import random
from random import choice
import datetime
from matplotlib.pylab import *
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from community import community_louvain
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns
from synthetic_data import *
from gl_sigrep import Gl_sigrep 
from primal_dual_gl import Primal_dual_gl 
from utils import *

node_num=20
signal_num=1000
error_sigma=0.01
knn_adj, knn_lap, knn_pos=knn_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)

rbf_dis=rbf_kernel(X_noise.T)
np.fill_diagonal(rbf_dis, 0)

Z=rbf_dis
Z=filter_graph_to_knn(Z, node_num)

alpha=3
beta=0.5
w_0=np.zeros(int((node_num-1)*node_num/2))
c=0
primal_gl=Primal_dual_gl(node_num, Z, alpha, beta, c=c)
vector_adj, primal_adj, error=primal_gl.run()


fig, (ax1, ax2)=plt.subplots(2,1, figsize=(5,8))
c1=ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('Ground Truth W')
c2=ax2.pcolor(primal_adj,cmap='RdBu')
ax2.set_title('Learned W')
fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
plt.show()


fig, (ax1, ax2)=plt.subplots(2,1, figsize=(5,8))
c1=ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('Ground Truth W')
c2=ax2.pcolor(primal_adj,cmap='RdBu')
ax2.set_title('filtered Learned W')
fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
plt.show()


plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Iteration', fontsize=12)
plt.show()