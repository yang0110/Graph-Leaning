import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import networkx as nx 
from sklearn.datasets import make_blobs
node_num=100
signal_num=100
error_sigma=0.1
scale=0.1
x, y=make_blobs(n_samples=node_num, n_features=2, centers=4, cluster_std=scale, center_box=(0,1), shuffle=False, random_state=20)
f=np.random.normal(size=(signal_num, 2))
signal=np.dot(f, x.T)

#noise=np.random.normal(size=(signal_num, node_num))
#signal=noise+signal

pos=x
adj_matrix=rbf_kernel(x)
np.fill_diagonal(adj_matrix,0)
filtered_adj_matrix=filter_graph_to_knn(adj_matrix, node_num, k=5)
graph=create_networkx_graph(node_num, filtered_adj_matrix)
edge_weight=adj_matrix[np.triu_indices(node_num, 1)]
edge_color=edge_weight[edge_weight>0]
nodes=nx.draw_networkx_nodes(graph, pos, node_color=signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()












