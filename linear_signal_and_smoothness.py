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

node_num=5
item_num=100
noise_scale=0.5
dimension=2
node_features=np.random.uniform(size=(node_num, dimension))
item_features=np.random.uniform(size=(item_num, dimension))
clear_signal=np.dot(item_features, node_features.T)
noise=np.random.normal(size=(item_num, node_num), scale=noise_scale)
noisy_signal=clear_signal+noise
nor=Normalizer()
clear_signal=nor.fit_transform(clear_signal)
noisy_signal=nor.fit_transform(noisy_signal)

adj=rbf_kernel(node_features)
np.fill_diagonal(adj,0)
adj[np.triu_indices(node_num,1)]=0
adj=adj+adj.T
d=np.sum(adj, axis=1)
degree=np.diag(d)
P=np.dot(np.linalg.inv(degree), adj)
lap=csgraph.laplacian(adj, normed=False)

evalues, evectors=np.linalg.eig(lap)
order=list(np.arange(0,node_num))
sorted_evectors=evectors[[x for _, x, in sorted(zip(evalues, order))]]
sorted_evalues=[x for x, _, in sorted(zip(evalues, order))]

smooth_signal=np.dot(clear_signal, P)
smooth_signal=nor.fit_transform(smooth_signal)

s_noisy=find_smoothness(noisy_signal[0], lap)
s_clear=find_smoothness(clear_signal[0], lap)
s_smooth=find_smoothness(smooth_signal[0],lap)

plt.plot([s_smooth, s_clear, s_noisy])
plt.show()

total_cum_energy1=0
total_cum_energy2=0
total_cum_energy3=0
for i in range(node_num):
	total_cum_energy1+=(node_num+1-i)*np.linalg.norm(np.dot(clear_signal, sorted_evectors[i].T))
	total_cum_energy2+=(node_num+1-i)*np.linalg.norm(np.dot(noisy_signal, sorted_evectors[i].T))
	total_cum_energy3+=(node_num+1-i)*np.linalg.norm(np.dot(smooth_signal, sorted_evectors[i].T))

plt.plot([total_cum_energy3, total_cum_energy1, total_cum_energy2])
plt.show()


def total_cum_energy(signal, eigen_vectors, node_num):
	for i in range(node_num):
		tce+=(node_num+1-i)*np.linalg.norm(np.dot(signal, eigen_vectors[i].T))
	return tce


def cum_energy(signal, eigen_vectors, node_num):
	ce_list=[]
	for i in range(node_num):
		ce=np.linalg.norm(np.dot(signal, eigen_vectors[:i].T))
		ce_list.extend([ce])
	return ce_list

cum_e_list1=[]
cum_e_list2=[]
cum_e_list3=[]
for i in range(node_num):
	cum_energy=np.linalg.norm(np.dot(noisy_signal,sorted_evectors[:i].T))
	cum_e_list1.extend([cum_energy])
	cum_energy=np.linalg.norm(np.dot(clear_signal,sorted_evectors[:i].T))
	cum_e_list2.extend([cum_energy])
	cum_energy=np.linalg.norm(np.dot(smooth_signal,sorted_evectors[:i].T))
	cum_e_list3.extend([cum_energy])

plt.plot(cum_e_list1, label='noisy')
plt.plot(cum_e_list2, label='clear')
plt.plot(cum_e_list3, label='smooth')
plt.legend(loc=1, fontsize=12)
plt.show()



plt.figure(figsize=(6,5))
plt.pcolor(adj, cmap=plt.cm.jet)
plt.colorbar()
plt.show()


graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.show()

clear_sample_signal=clear_signal[0]
noise_sample_signal=noisy_signal[0]
graph=create_networkx_graph(node_num, adj)
edge_weight=adj[np.triu_indices(node_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, node_features, node_size=100, node_color=clear_sample_signal, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, node_features, width=1.0, alpha=0.5, edge_color=edge_color, edge_cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.show()


alpha=1
beta=0.2
step_size=0.5
theta=10
test_signal=clear_signal
Z=euclidean_distances(test_signal.T, squared=True)
np.fill_diagonal(Z,0)
Z=norm_W(Z, node_num)
primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
primal_adj, error=primal_gl.run()
lap=csgraph.laplacian(primal_adj, normed=False)
signal=np.dot(test_signal, np.linalg.inv((np.identity(node_num)+theta*lap)))

pos=node_features
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=test_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=signal[0], cmap=plt.cm.jet)
plt.show()

def learn_graph(signal, node_num, alpha, beta, step_size):
	Z=euclidean_distances(signal.T, squared=True)
	np.fill_diagonal(Z,0)
	Z=norm_W(Z, node_num)
	primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
	primal_adj, error=primal_gl.run()
	lap=csgraph.laplacian(primal_adj, normed=False)
	return primal_adj, lap

def learn_signal_Tikhonov(signal, adj, lap, theta, node_num):
	signal=np.dot(signal, np.linalg.inv((np.identity(node_num)+theta*lap)))
	return signal

def learn_signal_TV(signal, adj, theta):
	gamma=1/theta
	d=pyunlocbox.functions.dummy()
	r=pyunlocbox.functions.norm_l1()
	f=pyunlocbox.functions.norm_l2(w=1, y=signal, lambda_=gamma)
	G = graphs.Graph(W=adj)
	G.compute_differential_operator()
	L=G.D.toarray()
	step=0.999/(1+np.linalg.norm(L))
	solver=pyunlocbox.solvers.mlfbf(L=L, step=step)
	x0=signal.copy()
	prob=pyunlocbox.solvers.solve([d,r,f], solver=solver, x0=x0, rtol=0, maxit=1000)
	solution=prob['sol']
	return solution

ls=learn_signal_TV(noisy_signal[0], adj, 0.1)

def learn_node_features_TV(signal, adj, theta, node_num, dimension, item_features):
	theta=3
	d=pyunlocbox.functions.dummy()
	r=pyunlocbox.functions.norm_l1()
	f=pyunlocbox.functions.norm_l2(w=1, y=signal.T, lambda_=theta)
	G = graphs.Graph(W=adj)
	G.compute_differential_operator()
	L=G.D.toarray()
	step=0.999/(1+np.linalg.norm(L))
	solver=pyunlocbox.solvers.mlfbf(L=L, step=step)
	x0=signal.copy().T
	prob=pyunlocbox.solvers.solve([d,r,f], solver=solver, x0=x0, rtol=0, maxit=10000)
	solution=prob['sol']
	return solution

sol=learn_node_features_TV(noisy_signal, adj, theta, node_num, dimension, item_features)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

a=check_symmetric(adj)