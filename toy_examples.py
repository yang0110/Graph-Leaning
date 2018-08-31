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
from gl_algorithms import *
from sklearn.datasets import make_moons, make_circles, make_classification,make_blobs
from sklearn.preprocessing import StandardScaler

path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 


n_samples=256
noise=0.1
iteration=100

G=graphs.Sensor(N=n_samples, distributed=True, seed=42)
G.compute_fourier_basis()
label_signal=np.copysign(np.ones(G.N),G.U[:,3])
#G.plot_signal(label_signal)
rs=np.random.RandomState(42)
M=rs.rand(G.N)
M=(M>0.1).astype(float)
sigma=0.1
subsampled_noisy_label_signal=M*(label_signal+sigma*rs.standard_normal(G.N))
#subsampled_noisy_label_signal=label_signal+np.random.normal(size=(n_samples))
#G.plot_signal(subsampled_noisy_label_signal)
x,y=G.coords, label_signal
y=subsampled_noisy_label_signal




#x,y=make_blobs(n_samples=n_samples, n_features=2, centers=4, cluster_std=0.1, center_box=(0,5),shuffle=False, random_state=0)

# linspace = np.linspace(0, 2 * np.pi, n_samples // 2 + 1)[:-1]
# outer_circ_x = 0.5*np.cos(linspace)
# outer_circ_y = 0.5*np.sin(linspace)
# inner_circ_x = outer_circ_x +3
# inner_circ_y = outer_circ_y 
# x = np.vstack((np.append(outer_circ_x, inner_circ_x),
#                    np.append(outer_circ_y, inner_circ_y))).T
# y = np.hstack([np.zeros(n_samples // 2, dtype=np.intp),
#                    np.ones(n_samples // 2, dtype=np.intp)])



#x,y=make_moons(n_samples=n_samples, noise=noise, random_state=0, shuffle=False)
#x,y=make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=0, shuffle=False)
x=StandardScaler().fit_transform(x)
adj=rbf_kernel(x)
np.fill_diagonal(adj, 0)
lap=csgraph.laplacian(adj, normed=False)
pos=x


newpath=path+'node_num_%s_noise_%s'%(n_samples, noise)+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

learned_graph=create_networkx_graph(n_samples, adj)
edge_color=adj[np.triu_indices(n_samples,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=label_signal, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.savefig(newpath+'True Signal'+'.png', dpi=200)
plt.clf()


learned_graph=create_networkx_graph(n_samples, adj)
edge_color=adj[np.triu_indices(n_samples,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=y, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.savefig(newpath+'Noisy Signal'+'.png', dpi=200)
plt.clf()




########################

k=n_samples
knn_adj=filter_graph_to_knn(adj, n_samples, k=k)
degree=np.sum(knn_adj, axis=1)
knn_D=np.diag(degree)
knn_y=np.dot(np.dot(np.linalg.inv(knn_D), knn_adj), y)

learned_graph=create_networkx_graph(n_samples, knn_adj)
edge_color=knn_adj[np.triu_indices(n_samples,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=knn_y, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.savefig(newpath+'Once_KNN_k_%s'%(k)+'.png', dpi=200)
plt.clf()


knn_y=y
knn_error_list=[]
knn_smoothness_list=[]
for i in range(iteration):
	print('i', i)
	knn_y=np.dot(np.dot(np.linalg.inv(knn_D), knn_adj), knn_y)
	knn_error=np.linalg.norm(knn_y-y)
	knn_error_list.extend([knn_error])
	knn_smoothness=np.dot(knn_y.T, np.dot(lap, knn_y))
	knn_smoothness_list.extend([knn_smoothness])

learned_graph=create_networkx_graph(n_samples, knn_adj)
edge_color=knn_adj[np.triu_indices(n_samples,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=knn_y, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.savefig(newpath+'converged_KNN_k_%s'%(k)+'.png', dpi=200)
plt.clf()

np.save(newpath+'knn_error_list'+'.npy', knn_error_list)
np.save(newpath+'knn_smoothness_list'+'.npy', knn_smoothness_list)

alpha_list=np.arange(0, 1, 0.01)
gl_error_list=[]
gl_smoothness_list=[]
for alpha in alpha_list:
	print('alpha', alpha)
	gl_y=np.dot(np.linalg.inv(np.identity(n_samples)+alpha*lap), y)

	gl_error=np.linalg.norm(gl_y-y)
	gl_error_list.extend([gl_error])
	gl_smoothness=np.dot(gl_y.T, np.dot(lap, gl_y))
	gl_smoothness_list.extend([gl_smoothness])

	learned_graph=create_networkx_graph(n_samples, adj)
	edge_color=adj[np.triu_indices(n_samples,1)]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=gl_y, node_size=100, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
	plt.axis('off')
	plt.savefig(newpath+'GL_alpha_%s'%(alpha)+'.png', dpi=200)
	plt.clf()

best_alpha=alpha_list[np.argmin(gl_error_list)]
gl_y=np.dot(np.linalg.inv(np.identity(n_samples)+best_alpha*lap), y)

learned_graph=create_networkx_graph(n_samples, adj)
edge_color=adj[np.triu_indices(n_samples,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=gl_y, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.savefig(newpath+'Best_GL_alpha_%s'%(best_alpha)+'.png', dpi=200)
plt.clf()

np.save(newpath+'gl_error_list'+'.npy', gl_error_list)
np.save(newpath+'gl_smooth_list'+'.npy', gl_smoothness_list)



k_list=np.arange(10, n_samples+10, 10)
once_knn_error_list=[]
once_knn_smooth_list=[]
for k in k_list:
	knn_adj=filter_graph_to_knn(adj, n_samples, k=k)
	knn_lap=csgraph.laplacian(knn_adj, normed=False)
	degree=np.sum(knn_adj, axis=1)
	knn_D=np.diag(degree)
	knn_y=np.dot(np.dot(np.linalg.inv(knn_D), knn_adj), y)
	error=np.linalg.norm(knn_y-y)
	smooth=np.dot(knn_y.T, np.dot(knn_lap, knn_y))
	once_knn_error_list.extend([error])
	once_knn_smooth_list.extend([smooth])


	learned_graph=create_networkx_graph(n_samples, knn_adj)
	edge_color=knn_adj[np.triu_indices(n_samples,1)]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=knn_y, node_size=100, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
	plt.axis('off')
	plt.savefig(newpath+'Once_KNN_k_%s'%(k)+'.png', dpi=200)
	plt.clf()

np.save(newpath+'once_knn_error_list'+'.npy', once_knn_error_list)
np.save(newpath+'once_knn_smooth_list'+'.npy', once_knn_smooth_list)


k_list=np.arange(10, n_samples+10, 10)
knn_y=y
converged_knn_error_list=[]
converged_knn_smooth_list=[]
for k in k_list:
	print('k',k)
	for ii in range(iteration):
		knn_adj=filter_graph_to_knn(adj, n_samples, k=k)
		knn_lap=csgraph.laplacian(adj, normed=False)

		degree=np.sum(knn_adj, axis=1)
		knn_D=np.diag(degree)
		knn_y=np.dot(np.dot(np.linalg.inv(knn_D), knn_adj), knn_y)
	error=np.linalg.norm(knn_y-y)
	smooth=np.dot(knn_y.T, np.dot(knn_lap, knn_y))
	converged_knn_error_list.extend([error])
	converged_knn_smooth_list.extend([smooth])

	learned_graph=create_networkx_graph(n_samples, knn_adj)
	edge_color=knn_adj[np.triu_indices(n_samples,1)]
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(learned_graph, pos, node_color=knn_y, node_size=100, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(learned_graph, pos, width=1.0, alpha=0.1, edge_color='grey')
	plt.axis('off')
	plt.savefig(newpath+'Converged_KNN_k_%s'%(k)+'.png', dpi=200)
	plt.clf()

np.save(newpath+'converged_knn_error_list'+'.npy', converged_knn_error_list)
np.save(newpath+'converged_knn_smooth_list'+'.npy', converged_knn_smooth_list)