import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import networkx as nx 
import os 
os.chdir('D:/Research/Graph Learning/code/')
from sklearn.metrics.pairwise import  rbf_kernel
import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import vector_form, sum_squareform
from pygsp import graphs, plotting

node_num=10
signal_num=100
error_signal=0.01
rbf_adj, rbf_lap, rbf_pos=rbf_graph(node_num)
knn_adj, knn_lap, knn_pos=knn_graph(node_num)
X, X_noise=generate_signal(signal_num, node_num, rbf_pos, error_signal)

G=create_networkx_graph(node_num, knn_adj)
edge_num=G.number_of_edges()
edge_weights=knn_adj[np.triu_indices(node_num)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(G, knn_pos, node_color=X[1,:], cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(G, knn_pos, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.show()

