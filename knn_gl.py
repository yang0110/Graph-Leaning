## Graph learning and signal learning 
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
import datetime
node_num=40
signal_num=100
error_sigma=0.1
adj_matrix, lap, pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, pos, error_sigma)


knn_adj, knn_lap=learn_knn_graph(X_noise, node_num, k=10)

plot_graph_and_signal(adj_matrix, X_noise[0], pos, node_num)
plot_graph_and_signal(knn_adj, X_noise[0], pos, node_num)
graph_error=np.linalg.norm(knn_adj-adj_matrix)














