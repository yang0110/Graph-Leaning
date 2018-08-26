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
from gl_algorithms import *
path='D:/Research/Graph Learning/code/results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=30
signal_num=100
error_sigma=0.1
scale=0.1
## graphs
blob_adj, blob_lap, blob_pos=blob_graph(node_num, scale=scale)
rbf_adj, rbf_lap, rbf_pos=rbf_graph(node_num, threshold=0.7)
knn_adj, knn_lap, knn_pos=knn_graph(node_num)

## linear signals
blob_signal=generate_signal(signal_num, node_num, blob_pos)
circle_signal=circle(signal_num, node_num, blob_pos)
rbf_signal=generate_signal(signal_num, node_num, rbf_pos)
knn_signal=generate_signal(signal_num, node_num, knn_pos)

##########################################################################################
real_adj=rbf_adj
signal=rbf_signal
pos=rbf_pos


signal_num=signal.shape[0]
mask=np.random.uniform(size=(signal_num, node_num))
mask=mask<0.9
error_sigma=0.1
noise=signal_noise(signal_num, node_num, scale=error_sigma)

noisy_signal=mask*signal
#noisy_signal=signal+noise

newpath=path+'node_num_%s_signal_num_%s_error_%s'%(node_num, signal_num, int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

iteration=5
p_adj, p_signal=Primal_dual_gl_loop(node_num, noisy_signal, iteration, alpha=1, beta=0.1, theta=0.01, step_size=0.05)


knn_signals={}
knn_adjs={}
graph_error=[]
signal_error=[]
x_ticks=[]
for k in np.arange(5,int(node_num*0.75),5):
	learned_knn_adj, learned_knn_lap=learn_knn_graph(noisy_signal, node_num, k=k)
	learned_knn_signal=learn_knn_signal(learned_knn_adj, noisy_signal, signal_num, node_num)
	knn_adjs[k]=learned_knn_adj
	knn_signals[k]=learned_knn_signal
	g_e=np.linalg.norm(learned_knn_adj-real_adj)
	s_e=np.linalg.norm(learned_knn_signal-signal)
	graph_error.extend([g_e])
	signal_error.extend([s_e])
	x_ticks.extend([k])


fig, (ax1, ax2)=plt.subplots(2,1, sharex=True)
ax1.plot(x_ticks, graph_error, label='Graph Error')
ax1.set_title('KNN Error')
ax1.set_ylabel('Graph Error')
ax2.plot(x_ticks, signal_error, label='Signal Error')
ax2.set_ylabel('Signal Error')
ax2.set_xlabel('K')
plt.show()


plt.plot(x_ticks, signal_error, label='signal error')
plt.plot(x_ticks, graph_error, label='graph error')
plt.title('KNN Signal Error')
plt.xlabel('K')
plt.legend(loc=1)
plt.show()





filtered_real_adj=filter_graph_to_knn(real_adj, node_num, k=10)
graph=create_networkx_graph(node_num, filtered_real_adj)
edge_weight=filtered_real_adj[np.triu_indices(node_num, 1)]
edge_color=edge_weight[edge_weight>0]
nodes=nx.draw_networkx_nodes(graph, pos, node_color=signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.title('True Signal')
plt.axis('off')
plt.show()


nodes=nx.draw_networkx_nodes(graph, pos, node_color=p_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.title('GL Signal')
plt.axis('off')
plt.show()

for k in np.arange(5,int(node_num*0.75),5):
	nodes=nx.draw_networkx_nodes(graph, pos, node_color=knn_signals[k][0], node_size=100, cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
	plt.title('KNN Signal K= %s'%(k))
	plt.axis('off')
	plt.show()


normed_real_adj=norm_W(real_adj, node_num)
normed_gl_adj=norm_W(p_adj, node_num)
normed_knn_adj=norm_W(learned_knn_adj, node_num)

fig, axes=plt.subplots(2,2)
axes[0,0].pcolor(normed_real_adj, cmap='jet')
axes[0,0].set_title('True')
axes[0,0].axis('off')
axes[0,1].axis('off')
axes[1,0].pcolor(normed_gl_adj, cmap='jet')
axes[1,0].set_title('GL')
axes[1,0].axis('off')
axes[1,1].pcolor(normed_knn_adj, cmap='jet')
axes[1,1].set_title('KNN')
axes[1,1].axis('off')
plt.show()



