## Graph learning and signal learning 
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('D:/Research/Graph Learning/code/')
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from gl_sigrep import Gl_sigrep
path='D:/Research/Graph Learning/code/results/test_results2/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=20
signal_num=100
error_sigma=0.1
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X_noise=generate_signal_gl_siprep(signal_num, node_num, knn_lap, error_sigma)
adj_matrix,knn_lap, knn_pos=knn_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)
#smooth_signal=f4(knn_pos[:,0], knn_pos[:,1])

newpath=path+'error_sigma_%s'%(int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signals=X_noise

signal_error_list=[]
graph_error_list=[]
z_error_list=[]
for i in range(10):
	print('i', i)
	#adj_matrix=filter_graph_to_knn(adj_matrix, node_num)
	rbf_dis=rbf_kernel(signals.T)
	np.fill_diagonal(rbf_dis, 0)
	Z=rbf_dis
	#Z=filter_graph_to_knn(rbf_dis, node_num)
	##graph learning 
	alpha=0.5
	beta=0.05
	primal_gl=Gl_sigrep(node_num, Z, alpha, beta)
	primal_adj, error=primal_gl.run(adj_matrix)

	##signal learning
	laplacian=csgraph.laplacian(primal_adj, normed=False)
	signals=np.dot(signals, np.linalg.inv((np.identity(node_num)+alpha*laplacian)))
	print('X', X)
	#print('x_noise', X_noise)
	print('signals', signals)
	signal_error=np.linalg.norm(signals-X)
	graph_error=np.linalg.norm(primal_adj-adj_matrix)
	z_error=np.linalg.norm(Z-adj_matrix)
	signal_error_list.extend([signal_error])
	graph_error_list.extend([graph_error])
	z_error_list.extend([z_error])
###plot

	real_signal=X_noise[0,:]
	learned_signal=signals[0,:]

	real_graph=create_networkx_graph(node_num, adj_matrix)
	edge_num=real_graph.number_of_edges()
	edge_weights=adj_matrix[np.triu_indices(node_num,0)]
	edge_color=edge_weights[edge_weights>0]
	edge_alpha=edge_color
	nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds)
	edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
	plt.axis('off')
	#plt.savefig(newpath+'real_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
	plt.show()

	### Learned graph and learned signal
	#primal_adj=filter_graph_to_knn(primal_adj, node_num)
	learned_graph=create_networkx_graph(node_num, primal_adj)
	edge_num=learned_graph.number_of_edges()
	edge_weights=primal_adj[np.triu_indices(node_num,0)]
	edge_color=edge_weights[edge_weights>0]
	edge_alpha=edge_color
	nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
	edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
	plt.axis('off')
	#plt.savefig(newpath+'learned_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
	plt.show()

	plt.plot(error)
	plt.ylabel('Learning Error', fontsize=12)
	plt.show()




plt.plot(signal_error_list)
plt.title('signal error', fontsize=12)
plt.show()

plt.plot(graph_error_list)
plt.title('graph error', fontsize=12)
plt.show()

plt.plot(z_error_list)
plt.title('z_error', fontsize=12)
plt.show()


############################################ Results
## Real Graph and real signal

real_signal=X_noise[0,:]
noise_signal=X_noise[0,:]
learned_signal=signals[0,:]

real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
#plt.savefig(newpath+'real_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()



### Learned graph and learned signal
#primal_adj=filter_graph_to_knn(primal_adj, node_num)
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
#plt.savefig(newpath+'learned_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()


fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('real W')
ax2.pcolor(primal_adj, cmap='RdBu')
ax2.set_title('learned w')
plt.show()





