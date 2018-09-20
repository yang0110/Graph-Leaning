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
from mpl_toolkits.mplot3d import Axes3D
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 


newpath='C:/Kaige_Research/Graph Learning/graph_learning_code/results/linear_signal_results/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

node_num=100
item_num=1000
noise_scale=0.1
dimension=2
alpha=1
beta=0.2
theta=0.1
step_size=0.5
item_features=np.random.uniform(size=(item_num, dimension))



G=graphs.Grid2d(N1=10, N2=10)
fig, axes=plt.subplots(1,2)
_=axes[0].spy(G.W)
G.plot(ax=axes[1])

adj=G.W.toarray()
pos=G.coords
node_features=pos.copy()
clear_signal=np.dot(item_features, node_features.T)

noisy_signal=clear_signal.copy()
noisy_signal2=clear_signal.copy()

for i in range(item_num):
	mask=choice(list(range(node_num)), size=10)
	noisy_signal[i, mask]=0
	mask=choice(list(range(node_num)), size=50)
	noisy_signal2[i, mask]=0


signal=clear_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.show()


lap=csgraph.laplacian(adj, normed=False)
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

cum_e_clear=cum_energy(clear_signal[0], sorted_evectors, node_num)
cum_e_noisy=cum_energy(noisy_signal[0], sorted_evectors, node_num)
cum_e_noisy2=cum_energy(noisy_signal2[0], sorted_evectors, node_num)

s1=find_smoothness(clear_signal[0], lap)
s2=find_smoothness(noisy_signal[0], lap)
s3=find_smoothness(noisy_signal2[0], lap)

plt.plot([s1, s2, s3], '*-')
plt.ylabel('Smoothness (Dirichlet Energy)', fontsize=12)
plt.savefig(newpath+'smoothnes'+'.png', dpi=100)
plt.show()

fig, (ax1, ax2, ax3)=plt.subplots(3,1, figsize=(5,5))
ax2.plot(np.diff(cum_e_noisy), label='noisy')
ax1.plot(np.diff(cum_e_clear), label='clear')
ax3.plot(np.diff(cum_e_noisy2), label='noisy2')
ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)
ax3.set_xlabel('Frequency', fontsize=12)
ax3.set_ylabel('Magitude', fontsize=12)
ax1.set_ylabel('Magitude', fontsize=12)
ax2.set_ylabel('Magitude', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'psd_signal'+'.png', dpi=100)
plt.show()

signal=clear_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'clear_signal'+'.png', dpi=100)
plt.show()

signal=noisy_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'noisy_signal'+'.png', dpi=100)
plt.show()

signal=noisy_signal2[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'noisy_signal2'+'.png', dpi=100)
plt.show()



## learn the graph 

Z=euclidean_distances(clear_signal.T, squared=True)
np.fill_diagonal(Z,0)
Z=norm_W(Z, node_num)

alpha_list=np.arange(0,3,0.1)
beta_list=np.arange(0,1,0.1)
error_list=[]
for alpha in alpha_list:
	for beta in beta_list:
		print('alpha, beta', alpha, beta)
		primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=0.5)
		primal_adj, error=primal_gl.run()
		primal_lap=csgraph.laplacian(primal_adj, normed=False)
		error=np.linalg.norm(primal_adj-adj)
		error_list.extend([error])


fig = plt.figure()
ax = fig.gca(projection='3d')
X = alpha_list
Y = beta_list
X, Y = np.meshgrid(X, Y)
Z=np.array(error_list).reshape(X.shape[1], X.shape[0]).T
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Alpha', fontsize=12)
ax.set_ylabel('Beta', fontsize=12)
ax.set_zlabel('graph error', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'gl_graph_error')
plt.show()

gamma_list=np.arange(0.01,1,0.1)
k_list=np.arange(1, 10, 1)
error_list=[]
for gamma in gamma_list:
	for k in k_list:
		print('gamme, k', gamma, k)
		knn_adj,knn_lap=learn_knn_graph(clear_signal, node_num, k=k, gamma=gamma)
		error=np.linalg.norm(knn_adj-adj)
		error_list.extend([error])

fig = plt.figure()
ax = fig.gca(projection='3d')
X = gamma_list
Y = k_list
X, Y = np.meshgrid(X, Y)
Z=np.array(error_list).reshape(X.shape[1], X.shape[0]).T
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Gamma', fontsize=12)
ax.set_ylabel('K', fontsize=12)
ax.set_zlabel('graph error', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'rbf_graph_error')
plt.show()

### denoise signal
d=np.sum(adj, axis=1)
D=np.diag(d)
P=np.dot(np.linalg.inv(D), adj)
denoise_signal=np.dot(noisy_signal,P)
denoise_signal2=np.dot(noisy_signal2,P)
smooth_signal=np.dot(noisy_signal, np.linalg.inv(np.identity(node_num)+theta*lap))
smooth_signal2=np.dot(noisy_signal2, np.linalg.inv(np.identity(node_num)+theta*lap))

signal=noisy_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'noisy_signal'+'.png', dpi=100)
plt.show()



signal=noisy_signal2[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'noisy_signal2'+'.png', dpi=100)
plt.show()

signal=denoise_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'denoised_signal'+'.png', dpi=100)
plt.show()

signal=denoise_signal2[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'denoised_signal2'+'.png', dpi=100)
plt.show()


signal=smooth_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'smooth_signal'+'.png', dpi=100)
plt.show()

signal=smooth_signal2[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=signal, cmap=plt.cm.jet, vmin=0, vmax=1)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.savefig(newpath+'smooth_signal2'+'.png', dpi=100)
plt.show()



cum_e_clear=cum_energy(noisy_signal2[0], sorted_evectors, node_num)
cum_e_noisy=cum_energy(denoise_signal2[0], sorted_evectors, node_num)
cum_e_noisy2=cum_energy(smooth_signal2[0], sorted_evectors, node_num)

s1=find_smoothness(noisy_signal2[0], lap)
s2=find_smoothness(denoise_signal2[0], lap)
s3=find_smoothness(smooth_signal2[0], lap)

plt.plot([s1, s2, s3], '*-')
plt.ylabel('Smoothness (Dirichlet Energy)', fontsize=12)
plt.savefig(newpath+'smoothnes'+'.png', dpi=100)
plt.show()

fig, (ax1, ax2, ax3)=plt.subplots(3,1, figsize=(5,5))
ax2.plot(np.diff(cum_e_noisy), label='noisy')
ax1.plot(np.diff(cum_e_clear), label='denoised')
ax3.plot(np.diff(cum_e_noisy2), label='smooth')
ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)
ax3.set_xlabel('Frequency', fontsize=12)
ax3.set_ylabel('Magitude', fontsize=12)
ax1.set_ylabel('Magitude', fontsize=12)
ax2.set_ylabel('Magitude', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'psd_signal2'+'.png', dpi=100)
plt.show()
