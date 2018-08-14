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
os.chdir('D:/Research/Graph Learning/code/')
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
from utils import vector_form, sum_squareform

node_num=10
signal_num=100
rbf_adj, rbf_lap, rbf_pos=rbf_graph(node_num)
X_t=generate_signal(signal_num, node_num, rbf_lap)
X=X_t.T

Z=rbf_kernel(X)

iteration=1000
alpha=0.5
beta=0.1
w_0=np.zeros(int((node_num-1)*node_num/2))
c=0

primal_gl=Primal_dual_gl(node_num, Z, alpha, beta, iteration, c=c)

vector_adj, primal_adj=primal_gl.run()







fig, (ax1, ax2)=plt.subplots(1,2, figsize=(8,4))
c1=ax1.pcolor(rbf_adj,cmap='RdBu', vmin=0, vmax=1)
ax1.set_title('Ground Truth W')
c2=ax2.pcolor(primal_adj,cmap='RdBu', vmin=0, vmax=1)
ax2.set_title('Learned W')
fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
plt.show()