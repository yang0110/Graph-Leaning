import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import seaborn as sns
sns.set_style("white")
from synthetic_data import *
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from primal_dual_gl import Primal_dual_gl 
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd
from gl_sigrep import Gl_sigrep
from gl_algorithms import *

newpath='C:/Kaige_Research/Graph Learning/graph_learning_code/results/MAB_models_results/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

node_num=20
signal_num=100
error_sigma=0.1
dimension=2
threshold=0.5
alpha=1
beta=0.2
theta=0.01
target_trace=node_num

pos=np.random.uniform(size=(node_num, dimension))
rbf_adj, rbf_lap=generate_rbf_graph(node_num, pos, threshold=threshold)
er_adj, er_lap=generate_er_graph(node_num)
ba_adj, ba_lap=generate_ba_graph(node_num)

rbf_noisy_signal, rbf_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap, error_sigma)
er_noisy_signal, er_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap, error_sigma)
ba_noisy_signal, ba_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap, error_sigma)

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=rbf_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=rbf_noisy_signal[0], cmap=plt.cm.jet)
plt.show()

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=er_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=er_noisy_signal[0], cmap=plt.cm.jet)
plt.show()

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=ba_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=ba_noisy_signal[0], cmap=plt.cm.jet)
plt.show()


true_signal=ba_signal
noisy_signal=ba_noisy_signal
true_lap=ba_lap
true_adj=ba_adj


primal_adj, primal_lap, primal_signal=Primal_dual_gl_loop(node_num, noisy_signal, 10)
gl_adj, gl_lap, gl_signal=Siprep_gl_loop(node_num, noisy_signal, 10)


fig, axes=plt.subplots(2,2)
axes[0,0].pcolor(true_adj, cmap=plt.cm.jet)
axes[1,0].pcolor(primal_adj, cmap=plt.cm.jet)
axes[1,1].pcolor(gl_adj, cmap=plt.cm.jet)
axes[0,0].set_title('True Adj', fontsize=12)
axes[1,0].set_title('Primal Adj', fontsize=12)
axes[1,1].set_title('GL Adj', fontsize=12)
plt.tight_layout()
plt.show()

fig, axes=plt.subplots(2,2)
axes[0,0].scatter(pos[:,0], pos[:,1], c=true_signal[0], cmap=plt.cm.jet)
axes[0,1].scatter(pos[:,0], pos[:,1], c=noisy_signal[0], cmap=plt.cm.jet)
axes[1,0].scatter(pos[:,0], pos[:,1], c=primal_signal[0], cmap=plt.cm.jet)
axes[1,1].scatter(pos[:,0], pos[:,1], c=gl_signal[0], cmap=plt.cm.jet)
axes[0,0].set_title('True Signal', fontsize=12)
axes[0,1].set_title('Noisy Signal', fontsize=12)
axes[1,0].set_title('Primal Signal', fontsize=12)
axes[1,1].set_title('GL Signal', fontsize=12)
plt.tight_layout()
plt.show()


graph=create_networkx_graph(node_num, true_adj)
edge_weight=true_adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=true_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('True Signal', fontsize=12)
plt.savefig(newpath+'True_Signal'+'.png', dpi=100)
plt.show()


graph=create_networkx_graph(node_num, true_adj)
edge_weight=true_adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=noisy_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('Noisy Signal', fontsize=12)
plt.savefig(newpath+'noisy_Signal'+'.png', dpi=100)
plt.show()



graph=create_networkx_graph(node_num, primal_adj)
edge_weight=primal_adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=primal_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('Primal Signal', fontsize=12)
plt.savefig(newpath+'Primal_Signal'+'.png', dpi=100)
plt.show()



graph=create_networkx_graph(node_num, gl_adj)
edge_weight=gl_adj[np.triu_indices(node_num,1)]
edge_color=edge_weight
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=gl_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('Gl Signal', fontsize=12)
plt.savefig(newpath+'gl_signal'+'.png', dpi=100)
plt.show()