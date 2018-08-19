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

node_num=5
signal_num=100
error_sigma=0.01
adj_matrix, knn_lap, knn_pos=knn_graph(node_num)
norm_adj_matrix=norm_W(adj_matrix, node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)
signals=X_noise

newpath=path+'error_sigma_%s'%(int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)


signal_error_list=[]
graph_error_list=[]
for i in range(10):
	print('i', i)
	Z=euclidean_distances(signals.T, squared=True)
	np.fill_diagonal(Z, 0)
	##graph learning 
	alpha=0.1  ## bigger alpha --- bigger weights
	beta=0.5   ### bigger beta --- more dense
	primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta)
	primal_adj, error=primal_gl.run(adj_matrix)
	norm_primal_adj=norm_W(primal_adj,node_num)
	laplacian=csgraph.laplacian(primal_adj, normed=False)
	signals=np.dot(signals, np.linalg.inv((np.identity(node_num)+laplacian)))

	#print('adj_matrix \n', adj_matrix)
	#print('norm_primal_adj \n', norm_primal_adj)
	#print('primal_adj \n', primal_adj)

	print('X\n', X[0,:])
	print('signals\n', signals[0,:])

	signal_error=np.linalg.norm(signals-X)
	graph_error=np.linalg.norm(primal_adj-adj_matrix)
	signal_error_list.extend([signal_error])
	graph_error_list.extend([graph_error])



plt.plot(signal_error_list)
plt.title('signal error', fontsize=12)
plt.show()

plt.plot(graph_error_list)
plt.title('graph error', fontsize=12)
plt.show()



real_signal=X[0,:]
learned_signal=signals[0,:]

real_graph=create_networkx_graph(node_num, adj_matrix)
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()


primal_adj=filter_graph_to_knn(primal_adj, node_num)
norm_primal_adj=norm_W(primal_adj, node_num)
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues,vmin=0, vmax=1)
plt.axis('off')
plt.show()


fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('real W')
ax2.pcolor(primal_adj, cmap='RdBu')
ax2.set_title('learned w')
plt.show()




