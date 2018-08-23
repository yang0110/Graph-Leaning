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
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/test_results2/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=20
signal_num=100
error_sigma=0.1
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)
#original_signal=original_signal(node_num)
#X=Heat_diffusion_signal(original_signal, adj_matrix)
#X=Tikhonov_signal(original_signal, adj_matrix)
#X=Generative_model_signal(original_signal, adj_matrix)

newpath=path+'error_%s'%(int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signals=X_noise

Z=euclidean_distances(signals.T, squared=True)
np.fill_diagonal(Z, 0)
Z=norm_W(Z, node_num)


alpha=1
beta=0.2
theta=0.01
primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta)
#primal_gl=Gl_sigrep(node_num, Z, alpha=alpha, beta=beta)
primal_adj, error=primal_gl.run(adj_matrix)

laplacian=csgraph.laplacian(primal_adj, normed=False)
learned_signals=np.dot(signals, np.linalg.inv((np.identity(node_num)+theta*laplacian)))


print('adj_matrix\n', adj_matrix)
print('primal_adj\n', primal_adj)

print('X\n', X[0,:])
print('learned_signals\n', learned_signals[0,:])

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


learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,1)]
edge_weights[edge_weights<0]=0
edge_color=edge_weights
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues,)
plt.axis('off')
plt.show()



##Plot graph 
fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('real W')
ax2.pcolor(primal_adj, cmap='RdBu')
ax2.set_title('learned w')
plt.show()



### plot graph learning error

plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.show()






# fig, axes=plt.subplots(3,2, figsize=(6,6))
# axes[1,0].pcolor(Z,cmap='RdBu')
# axes[1,0].axis('off')
# axes[1,1].pcolor(rbf_z,cmap='RdBu')
# axes[1,1].axis('off')
# axes[0,0].pcolor(adj_matrix,cmap='RdBu')
# axes[0,0].axis('off')
# axes[2,0].pcolor(primal_adj,cmap='RdBu')
# axes[2,0].axis('off')
# axes[2,1].pcolor(primal_adj_z,cmap='RdBu')
# axes[2,1].axis('off')
# axes[1,0].set_title('Z')
# axes[1,1].set_title('rbf_z')
# axes[0,0].set_title('adj_matrix')
# axes[2,0].set_title('primal_adj')
# axes[2,1].set_title('primal_adj_z')
# axes[0,1].axis('off')
# plt.show()
