import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('D:/Research/Graph Learning/code/')
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

newpath='D:/Research/Graph Learning/code/results/MAB_models_results/gl_and_primal_results/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

node_num=100
signal_num=100
error_sigma=0.1
dimension=2
threshold=0.5
alpha=1
beta=0.2
step_size=0.5
theta=0.01
target_trace=node_num

pos=np.random.uniform(size=(node_num, dimension))
rbf_adj, rbf_lap=generate_rbf_graph(node_num, pos, threshold=threshold)
er_adj, er_lap=generate_er_graph(node_num)
ba_adj, ba_lap=generate_ba_graph(node_num)
rbf_noisy_signal, rbf_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap, error_sigma)
er_noisy_signal, er_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap, error_sigma)
ba_noisy_signal, ba_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap, error_sigma)

def find_smoothness(signal, lap):
	smoothness=np.dot(signal, np.dot(lap, signal.T))
	return smoothness

signal=rbf_signal[0]
plt.scatter(pos[:,0], pos[:,1], c=signal, cmap=plt.cm.jet)
plt.show()

d=np.sum(rbf_adj, axis=1)
rbf_d_m=np.diag(d)
original_smoothness=find_smoothness(signal, rbf_lap)

iteration=10
smoothness_list=[]
for i in range(iteration):
	smooth_signal=np.dot(np.dot(np.linalg.inv(rbf_d_m), rbf_adj), signal)
	smoothness=find_smoothness(smooth_signal, rbf_lap)
	signal=smooth_signal
	smoothness_list.extend([smoothness])
#smoothness_list=[original_smoothness]+smoothness_list

plt.scatter(pos[:,0], pos[:,1], c=smooth_signal, cmap=plt.cm.jet)
plt.show()

plt.plot(smoothness_list)
plt.show()





true_signal=rbf_signal
noisy_signal=rbf_noisy_signal
true_lap=rbf_lap
true_adj=rbf_adj


Z=euclidean_distances(noisy_signal.T, squared=True)
np.fill_diagonal(Z,0)
Z=norm_W(Z, node_num)

alpha_list=np.arange(0) ## scale
beta_list=np.arange(0)## sparseness
graph_error_list=[]
smooth_list=[]
for alpha in alpha_list:
	primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
	primal_adj, error=primal_gl.run()
	primal_lap=csgraph.laplacian(primal_adj, normed=False)
	graph_error=np.linalg.norm(primal_adj-true_adj)
	smooth=np.dot(noisy_signal[0], np.dot(primal_lap, noisy_signal[0].T))
	graph_error_list.extend([graph_error])
	smooth_list.extend([smooth])

knn_adj, knn_lap=learn_knn_graph(noisy_signal, node_num, k=node_num)

true_smooth=np.dot(noisy_signal[0], np.dot(true_lap, noisy_signal[0].T))
knn_graph_error=np.linalg.norm(knn_adj-true_adj)

graph_error_list=[knn_graph_error]+graph_error_list
smooth_list=[true_smooth]+smooth_list
ticks=[-0.1]+list(alpha_list)

fig, (ax1, ax2)=plt.subplots(2,1)
ax1.plot(ticks, graph_error_list,  '.', label='Graph Error')
ax1.set_ylabel('Graph Error', fontsize=12)
ax1.legend(loc=1)
ax2.plot(ticks, smooth_list, '.', label='Smoothness')
ax2.set_xlabel('Alpha', fontsize=12)
ax2.set_ylabel('Smoothness', fontsize=12)

ax2.legend(loc=1)
plt.tight_layout()
plt.savefig(newpath+'graph_error_and_smoothness_vs_alpha')
plt.show()



# primal_lap=csgraph.laplacian(primal_adj, normed=False)
# primal_signal=np.dot(noisy_signal, np.linalg.inv((np.identity(node_num)+theta*primal_lap)))
# knn_adj=filter_graph_to_knn(knn_adj, node_num, k=k)
# knn_lap=csgraph.laplacian(knn_adj, normed=False)
# knn_signal=learn_knn_signal(knn_adj, noisy_signal, signal_num, node_num)



