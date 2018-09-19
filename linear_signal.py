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
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
import pyunlocbox


newpath='C:/Kaige_Research/Graph Learning/graph_learning_code/results/MAB_models_results/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

node_num=100
item_num=10
noise_scale=0.0
dimension=2
alpha=1
beta=0.2
theta=0.4
step_size=0.5
node_features=np.random.uniform(size=(node_num, dimension))
item_features=np.random.uniform(size=(item_num, dimension))
clear_signal=np.dot(item_features, node_features.T)
noise=np.random.normal(size=(item_num, node_num), scale=noise_scale)
noisy_signal=clear_signal+noise

for i in range(item_num):
	mask=choice(list(range(node_num)), size=10)
	noisy_signal[i,mask]=0

# pos=np.random.uniform(size=(node_num, dimension))
# rbf_adj, rbf_lap=generate_rbf_graph(node_num, pos, threshold=0.5)
# er_adj, er_lap=generate_er_graph(node_num)
# ba_adj, ba_lap=generate_ba_graph(node_num)


# adj=er_adj.copy()
# lap=csgraph.laplacian(adj, normed=True)
# d=np.sum(adj, axis=1)
# true_d_m=np.diag(d)
# P=np.dot(np.linalg.inv(true_d_m), adj)
# noisy_signal=np.random.normal(size=(item_num, node_num))

#nor=Normalizer()
#noisy_signal=nor.fit_transform(noisy_signal)

#nor=Normalizer()
#clear_signal=nor.fit_transform(clear_signal)

adj=rbf_kernel(node_features)
adj[adj<0.5]=0
np.fill_diagonal(adj,0)
adj[np.triu_indices(node_num,1)]=0
adj=adj+adj.T
d=np.sum(adj, axis=1)
degree=np.diag(d)
P=np.dot(np.linalg.inv(degree), adj)
lap=csgraph.laplacian(adj, normed=False)

iteration=5
smooth_signal=noisy_signal.copy()
signal_error=[]
for i in range(iteration):
	print('i', i)
	smooth_signal=np.dot(P, smooth_signal.T).T
	se=np.linalg.norm(smooth_signal-clear_signal)
	signal_error.extend([se])

#nor=Normalizer()
#smooth_signal=nor.fit_transform(smooth_signal)

evalues, evectors=np.linalg.eig(lap)
order=list(np.arange(0,node_num))
sorted_evectors=evectors[[x for _, x, in sorted(zip(evalues, order))]]
sorted_evalues=[x for x, _, in sorted(zip(evalues, order))]

def cum_energy(signal, eigen_vectors, node_num):
	ce_list=[]
	for i in range(node_num):
		ce=np.linalg.norm(np.dot(signal, eigen_vectors[:i].T))
		ce_list.extend([ce])
	return ce_list

cum_e_clear=cum_energy(clear_signal, sorted_evectors, node_num)
cum_e_noisy=cum_energy(noisy_signal, sorted_evectors, node_num)
cum_e_smooth=cum_energy(smooth_signal, sorted_evectors, node_num)

plt.plot(cum_e_clear, label='clear', marker='o')
plt.plot(cum_e_noisy, label='noisy')
plt.plot(cum_e_smooth, label='smooth')
plt.legend(loc=1)
plt.show()

plt.plot(np.diff(cum_e_clear), label='clear', marker='o')
plt.plot(np.diff(cum_e_noisy), label='noisy')
plt.plot(np.diff(cum_e_smooth), label='smooth')
plt.legend(loc=1)
plt.show()
plt.plot(signal_error)
plt.show()

graph_error_list=[]
signal_error_list=[]
smoothness_list=[]
learned_signals=[]
for i in range(item_num):
	print('i',i)
	smooth=[]
	signal=clear_signal[i]
	graph=create_networkx_graph(node_num, adj)
	edge_weight=adj[np.triu_indices(node_num,1)]
	edge_color=edge_weight[edge_weight>0]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
	plt.axis('off')
	plt.title('True Graph', fontsize=12)
	plt.show()

	s=find_smoothness(signal, lap)
	smooth.extend([s])

	signal=noisy_signal[i]
	graph=create_networkx_graph(node_num, adj)
	edge_weight=adj[np.triu_indices(node_num,1)]
	edge_color=edge_weight[edge_weight>0]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
	plt.axis('off')
	plt.title('Noisy signal', fontsize=12)
	plt.show()

	s=find_smoothness(signal, lap)
	smooth.extend([s])
######
	Z=euclidean_distances(signal.reshape(-1,1), squared=True)
	np.fill_diagonal(Z,0)
	Z=norm_W(Z, node_num)
	primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
	primal_adj, error=primal_gl.run()
	lap=csgraph.laplacian(primal_adj, normed=False)
	learned_signal=np.dot(signal, np.linalg.inv((np.identity(node_num)+theta*lap)))
	ge=np.linalg.norm(primal_adj-adj)
	graph_error_list.extend([ge])
	se=np.linalg.norm(learned_signal-clear_signal[i])
	signal_error_list.extend([se])

	s=find_smoothness(learned_signal, lap)
	smooth.extend([s])

	signal=learned_signal
	graph=create_networkx_graph(node_num, adj)
	edge_weight=adj[np.triu_indices(node_num,1)]
	edge_color=edge_weight[edge_weight>0]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
	plt.axis('off')
	plt.title('Learned Graph', fontsize=12)
	plt.show()
	smoothness_list.append(smooth)
	learned_signals.append(learned_signal)


smooth_signal=nor.fit_transform(np.array(learned_signals))
smooth_signal=np.dot(smooth_signal, P)
cum_e_clear=cum_energy(clear_signal, sorted_evectors, node_num)
cum_e_noisy=cum_energy(noisy_signal, sorted_evectors, node_num)
cum_e_smooth=cum_energy(smooth_signal, sorted_evectors, node_num)

plt.plot(cum_e_clear, label='clear')
plt.plot(cum_e_noisy, label='noisy')
plt.plot(cum_e_smooth, label='smooth')
plt.legend(loc=1)
plt.show()

plt.plot(np.diff(cum_e_clear), label='clear')
plt.plot(np.diff(cum_e_noisy), label='noisy')
plt.plot(np.diff(cum_e_smooth), label='smooth')
plt.legend(loc=1)
plt.show()

plt.plot(graph_error_list)
plt.title('graph error')
plt.show()

plt.plot(signal_error_list)
plt.title('signal error')
plt.show()

plt.plot(smoothness_list)
plt.legend(loc=1)
plt.show()