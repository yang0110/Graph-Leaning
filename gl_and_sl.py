## Graph learning and signal learning 
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel
#import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
node_num=100
signal_num=1000
error_sigma=0.1
adj_matrix, knn_lap, knn_pos=knn_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)
#adj_matrix, node_pos, X, X_noise=make_blobs_signal(signal_num, node_num, 4, 2, error_sigma)

adj_matrix=filter_graph_to_knn(adj_matrix, node_num)
rbf_dis=rbf_kernel(X_noise.T)
np.fill_diagonal(rbf_dis, 0)
Z=rbf_dis
Z=filter_graph_to_knn(Z, node_num)
noise_signal=X_noise[0,:]

alpha=5
beta=0.1
w_0=np.zeros(int((node_num-1)*node_num/2))
c=0
primal_gl=Primal_dual_gl(node_num, Z, alpha, beta, c=c)
vector_adj, primal_adj, error=primal_gl.run()


primal_adj=filter_graph_to_knn(primal_adj, node_num)
gamma = 3
G=graphs.Graph(primal_adj)
G.compute_differential_operator()
L = G.D.toarray()
d = pyunlocbox.functions.dummy()
r = pyunlocbox.functions.norm_l2(A=L, tight=False)
f = pyunlocbox.functions.norm_l2(w=1, y=noise_signal.copy(), lambda_=gamma)

step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma, 2)
solver = pyunlocbox.solvers.gradient_descent(step=step)
x0 = noise_signal.copy()
prob2 = pyunlocbox.solvers.solve([r, f], solver=solver,
                                x0=x0, rtol=0, maxit=1000)
learned_signal=prob2['sol']



############################################ Results
## Real Graph and real signal
real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
signal_ori=X[0,:]
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=signal_ori,node_size=100, cmap=plt.cm.Reds, vmin=np.min(signal_ori), vmax=np.max(signal_ori))
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.axis('off')
plt.show()


### Learned graph and learned signal
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds, vmin=np.min(signal_ori), vmax=np.max(signal_ori))
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.axis('off')
plt.show()



##Plot graph 

fig, (ax1, ax2)=plt.subplots(2,1, figsize=(5,8))
c1=ax1.pcolor(adj_matrix, cmap='RdBu', vmin=0, vmax=1)
ax1.set_title('Ground Truth W')
c2=ax2.pcolor(primal_adj, cmap='RdBu', vmin=0, vmax=1)
ax2.set_title('Learned W')
fig.colorbar(c1, ax=ax1)
fig.colorbar(c2, ax=ax2)
plt.show()

### plot signal

plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.show()


