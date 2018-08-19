## Graph learning and signal learning 
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from gl_sigrep import Gl_sigrep
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/test_results2/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=10
signal_num=100
error_sigma=0.1
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)

newpath=path+'error_%s'%(int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signals=X_noise

Z=euclidean_distances(signals.T, squared=True)
np.fill_diagonal(Z, 0)

alpha=1
beta=1
primal_gl=Primal_dual_gl(node_num, Z, alpha, beta)
primal_adj, error=primal_gl.run(adj_matrix)
laplacian=csgraph.laplacian(primal_adj, normed=False)
learned_signals=np.dot(signals, np.linalg.inv((np.identity(node_num)+laplacian)))

print('adj_matrix\n', adj_matrix)
print('primal_adj\n', primal_adj)

print('X\n', X[1,:])
print('learned_signals\n', learned_signals[1,:])

signal_error=np.linalg.norm(learned_signals-X)
graph_error=np.linalg.norm(primal_adj-adj_matrix)

############################################ Results
## Real Graph and real signal
j=0
real_signal=X[j,:]
learned_signal=learned_signals[j,:]

real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()

#primal_adj=filter_graph_to_rbf(primal_adj, node_num)
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vim=0, vmax=1)
plt.axis('off')
plt.show()



##Plot graph 
fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('real W')
ax2.pcolor(norm_W(primal_adj, node_num), cmap='RdBu')
ax2.set_title('learned w')
plt.show()



### plot graph learning error

plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.show()


