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
from mpl_toolkits.mplot3d import Axes3D

newpath='C:/Kaige_Research/Graph Learning/graph_learning_code/results/MAB_models_results/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

node_num=20
signal_num=100
error_sigma=0.1
dimension=2
threshold=0.75
alpha=1
beta=0.2
step_size=0.5
theta=0.01
target_trace=node_num

pos=np.random.uniform(size=(node_num, dimension))
rbf_adj, rbf_lap=generate_rbf_graph(node_num, pos, threshold=threshold)
er_adj, er_lap=generate_er_graph(node_num)
ba_adj, ba_lap=generate_ba_graph(node_num)

true_adj=ba_adj
true_lap=ba_lap
d=np.sum(ba_adj, axis=1)
ba_d_m=np.diag(d)

signal=np.random.normal(size=(signal_num, node_num))
iteration=10
for i in range(iteration):
	signal=np.dot(np.dot(np.linalg.inv(ba_d_m), ba_adj), signal.T).T


test_signal=signal[0]
true_smooth=find_smoothness(test_signal, true_lap)
true_edge_num=len(true_adj[np.triu_indices(node_num, 1)])
true_edge_weight=np.sum(true_adj)
Z=euclidean_distances(signal.T, squared=True)
np.fill_diagonal(Z,0)
Z=norm_W(Z, node_num)

alpha_list=np.arange(0, 1, 0.1)
beta_list=np.arange(0, 1, 0.1)
s_list=[]
ge_list=[]
edge_num_list=[]
edge_weight_list=[]
for alpha in alpha_list:
	for beta in beta_list:
		print('alpha, beta', alpha, beta)
		primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=0.5)
		primal_adj, error=primal_gl.run()
		primal_lap=csgraph.laplacian(primal_adj, normed=False)
		graph_error=np.linalg.norm(primal_adj-true_adj)
		ge_list.extend([graph_error])
		sm=find_smoothness(test_signal, primal_lap)
		s_list.extend([sm])
		edge_num=len(primal_adj[np.triu_indices(node_num,1)])
		edge_weight=np.sum(primal_adj)
		edge_weight_list.extend([edge_weight])
		edge_num_list.extend([edge_num])


fig = plt.figure()
ax = fig.gca(projection='3d')
X = alpha_list
Y = beta_list
X, Y = np.meshgrid(X, Y)
Z=np.array(s_list).reshape(X.shape[0], X.shape[1])
Z2=true_smooth*np.ones((X.shape[0], X.shape[1]))
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, Z2, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Alpha', fontsize=12)
ax.set_ylabel('Beta', fontsize=12)
ax.set_zlabel('Smoothness', fontsize=12)
fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.savefig(newpath+'smooth_vs_alpha_beta')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = alpha_list
Y = beta_list
X, Y = np.meshgrid(X, Y)
Z=np.array(ge_list).reshape(X.shape[0], X.shape[1])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Alpha', fontsize=12)
ax.set_ylabel('Beta', fontsize=12)
ax.set_zlabel('graph Error', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'graph_error_vs_alpha_beta')

plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
X = alpha_list
Y = beta_list
X, Y = np.meshgrid(X, Y)
Z=np.array(edge_num_list).reshape(X.shape[0], X.shape[1])
Z2=true_edge_num*np.ones((X.shape[0], X.shape[1]))
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z2, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Alpha', fontsize=12)
ax.set_ylabel('Beta', fontsize=12)
ax.set_zlabel('Edge Num', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'edge_num_vs_alpha_beta')
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = alpha_list
Y = beta_list
X, Y = np.meshgrid(X, Y)
Z=np.array(edge_weight_list).reshape(X.shape[0], X.shape[1])
Z2=true_edge_weight*np.ones((X.shape[0], X.shape[1]))
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z2, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Alpha', fontsize=12)
ax.set_ylabel('Beta', fontsize=12)
ax.set_zlabel('Edge Weight', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'edge_weight_vs_alpha_beta')
plt.show()


gamma_list=np.arange(0, 3, 0.01)
k_list=np.arange(5, node_num, 1).astype(int)
s_list=[]
ge_list=[]
edge_num_list=[]
edge_weight_list=[]
for gamma in gamma_list:
	for k in k_list:
		print('gamma,k', gamma, k)
		knn_adj, knn_lap=learn_knn_graph(signal, node_num, k=k, gamma=gamma)
		graph_error=np.linalg.norm(knn_adj-true_adj)
		ge_list.extend([graph_error])
		sm=find_smoothness(test_signal, knn_lap)
		s_list.extend([sm])
		edge_num=len(knn_adj[np.triu_indices(node_num,1)])
		edge_weight=np.sum(knn_adj)
		edge_weight_list.extend([edge_weight])
		edge_num_list.extend([edge_num])



fig = plt.figure()
ax = fig.gca(projection='3d')
X = gamma_list
Y = k_list
X, Y = np.meshgrid(X, Y)
Z=np.array(s_list).reshape(X.shape[0], X.shape[1])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('gamma', fontsize=12)
ax.set_ylabel('k', fontsize=12)
ax.set_zlabel('Smoothness', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'smooth_vs_K_gamma')

plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = gamma_list
Y = k_list
X, Y = np.meshgrid(X, Y)
Z=np.array(ge_list).reshape(X.shape[0], X.shape[1])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('gamma', fontsize=12)
ax.set_ylabel('k', fontsize=12)
ax.set_zlabel('graph Error', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'graph_error_vs_K_gamma')

plt.show()



fig = plt.figure()
ax = fig.gca(projection='3d')
X = gamma_list
Y = k_list
X, Y = np.meshgrid(X, Y)
Z=np.array(edge_num_list).reshape(X.shape[0], X.shape[1])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('gamma', fontsize=12)
ax.set_ylabel('k', fontsize=12)
ax.set_zlabel('Edge Num', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'edge_num_vs_K_gamma')

plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
X = gamma_list
Y = k_list
X, Y = np.meshgrid(X, Y)
Z=np.array(edge_weight_list).reshape(X.shape[0], X.shape[1])
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,
                       linewidth=0, antialiased=False)
ax.set_xlabel('gamma', fontsize=12)
ax.set_ylabel('k', fontsize=12)
ax.set_zlabel('Edge Weight', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(newpath+'edge_weight_vs_K_gamma')

plt.show()



# rbf_adj=norm_W(rbf_adj, node_num)
# primal_adj=norm_W(primal_adj, node_num)
# knn_adj=norm_W(knn_adj, node_num)

# fig, axes=plt.subplots(2,2)
# axes[0,0].pcolor(ba_adj, cmap=plt.cm.jet)
# axes[1,0].pcolor(knn_adj, cmap=plt.cm.jet)
# axes[1,1].pcolor(primal_adj, cmap=plt.cm.jet)
# axes[0,0].set_title('True Adj')
# axes[1,0].set_title('RBF Adj')
# axes[1,1].set_title('GL Adj')
# plt.tight_layout()
# plt.show()




# primal_lap=csgraph.laplacian(primal_adj, normed=False)
# primal_signal=np.dot(noisy_signal, np.linalg.inv((np.identity(node_num)+theta*primal_lap)))
# knn_adj=filter_graph_to_knn(knn_adj, node_num, k=k)
# knn_lap=csgraph.laplacian(knn_adj, normed=False)
# knn_signal=learn_knn_signal(knn_adj, noisy_signal, signal_num, node_num)



