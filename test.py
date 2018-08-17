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
import datetime
path='D:/Research/Graph Learning/code/results/test_results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=20
signal_num=1000
error_sigma=0.5
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)
adj_matrix=filter_graph_to_knn(adj_matrix, node_num)

newpath=path+'error_sigma_%s'%(error_sigma)+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signal_error_list=[]
graph_error_list=[]
knn_graph_error_list=[]
knn_signal_error_list=[]
cov_matrix={}
bias={}
node_f=np.zeros((node_num, 2))
node_f_avg=np.zeros((node_num,2))
for ii in range(node_num):
	cov_matrix[ii]=np.zeros((2,2))
	bias[ii]=np.zeros(2)


for i in np.arange(100)+1:
	print('i',i)
	data=X_noise[:i,:]
	noise_signal=X_noise[i,:]
	real_signal=X[i,:]
	rbf_dis=rbf_kernel(data.T)
	np.fill_diagonal(rbf_dis, 0)
	Z=filter_graph_to_knn(rbf_dis, node_num)
	## Learn Graph
	alpha=1
	beta=0.05
	primal_gl=Primal_dual_gl(node_num, Z, alpha, beta)
	primal_adj, error=primal_gl.run()

	## Learn signal
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
	                                x0=x0, rtol=0, maxit=2000)
	learned_signal=prob2['sol']




	signal_error=np.linalg.norm(learned_signal-real_signal)
	graph_error=np.linalg.norm(primal_adj-adj_matrix)
	signal_error_list.extend([signal_error])
	graph_error_list.extend([graph_error])
	### KNN graph construction
	knn_adj=rbf_kernel(node_f)
	np.fill_diagonal(knn_adj, 0)
	knn_adj=filter_graph_to_knn(knn_adj, node_num)
	knn_graph_error=np.linalg.norm(knn_adj-adj_matrix)
	knn_graph_error_list.extend([knn_graph_error])

	#KK signal learning 
	for j in range(node_num):
		cov_matrix[j]+=np.outer(item_features[i], item_features[i])
		bias[j]+=item_features[i]*noise_signal[j]
		node_f[j]=np.dot(np.linalg.inv(cov_matrix[j]), bias[j])
		knn_adj=rbf_kernel(node_f)
		for jj in range(node_num):
			row=knn_adj[jj,:].copy()
			k=10
			neighbors=np.argsort(row)[node_num-k:]
			node_f_avg[jj]=np.mean(node_f[neighbors], axis=0)
		knn_signal=np.dot(node_f_avg, item_features[i])
		knn_signal_error=np.linalg.norm(knn_signal-real_signal)
	knn_signal_error_list.extend([knn_signal_error])
	knn_adj=filter_graph_to_knn(knn_adj, node_num)




plt.plot(signal_error_list, label='GL')
plt.plot(knn_signal_error_list, label='KNN')
plt.ylabel('Signal Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.savefig(newpath+'signal_error'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()



plt.plot(graph_error_list, label='GL')
plt.plot(knn_graph_error_list, label='KNN')
plt.ylabel('Graph Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.savefig(newpath+'graph_error'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()






############################################ Results
## Real Graph and real signal
real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds, vmin=np.min(real_signal), vmax=np.max(real_signal))
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(newpath+'real_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)

plt.show()


### Learned graph and learned signal
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds, vmin=np.min(real_signal), vmax=np.max(real_signal))
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(newpath+'learned_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)

plt.show()


###knn 
knn_graph=create_networkx_graph(node_num, knn_adj)
edge_num=knn_graph.number_of_edges()
edge_weights=knn_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(knn_graph, knn_pos, node_color=knn_signal,node_size=100, cmap=plt.cm.Reds, vmin=np.min(real_signal), vmax=np.max(real_signal))
edges=nx.draw_networkx_edges(knn_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.axis('off')
plt.savefig(newpath+'knn_graph_and_signal'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)

plt.show()

##Plot w
plt.figure(figsize=(5,5))
plt.pcolor(adj_matrix, cmap='RdBu', vmin=np.min(adj_matrix), vmax=np.max(adj_matrix))
plt.colorbar()
plt.title('Real Adjacency Matrix')
plt.savefig(newpath+'real_w'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.pcolor(primal_adj, cmap='RdBu', vmin=np.min(adj_matrix), vmax=np.max(adj_matrix))
plt.colorbar()
plt.title('Learned Adjacency Matrix')
plt.savefig(newpath+'learned_w'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.pcolor(knn_adj, cmap='RdBu', vmin=np.min(adj_matrix), vmax=np.max(adj_matrix))
plt.colorbar()
plt.title('Knn Adjacency Matrix')
plt.savefig(newpath+'Knn_w'+'error_sigma_%s'%(error_sigma)+'.png', dpi=100)
plt.show()



plt.plot(error)
plt.ylabel('Learning Error', fontsize=12)
plt.show()


