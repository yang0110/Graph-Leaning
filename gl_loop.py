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

node_num=20
signal_num=100
error_sigma=0.1
adj_matrix, knn_lap, knn_pos=rbf_graph(node_num)
X, X_noise, item_features=generate_signal(signal_num, node_num, knn_pos, error_sigma)

newpath=path+'error_sigma_%s'%(int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

signals=X_noise

signal_error_reference=[]
signal_error_list=[]
graph_error_list=[]
trace1=[]
trace2=[]
trace3=[]
trace4=[]
trace5=[]
smoothness=[]
primal_adj=np.identity(node_num)
for i in range(5):
	print('i', i)
	Z=euclidean_distances(signals.T, squared=True)
	np.fill_diagonal(Z, 0)
	Z=norm_W(Z, node_num)

	##graph learning 
	alpha=1## bigger alpha --- bigger weights
	beta=0.2  ### bigger beta --- more dense ## For GL_sigrep beta is not used.
	theta=0.01
	#primal_gl=Gl_sigrep(node_num, Z, alpha=alpha, beta=beta, step_size=0.5)
	primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=0.01)
	primal_adj, error=primal_gl.run(adj_matrix)
	laplacian=csgraph.laplacian(primal_adj, normed=False)
	signals=np.dot(signals, np.linalg.inv((np.identity(node_num)+theta*laplacian)))
	smooth=calculate_smoothness(signals, laplacian)
	smoothness.append(smooth)


	#print('adj_matrix \n', adj_matrix)
	#print('primal_adj \n', primal_adj)

	print('X\n', X[0,:])
	print('signals\n', signals[0,:])

	signal_error_ref=np.linalg.norm(X_noise-X)
	signal_error=np.linalg.norm(signals-X)
	graph_error=np.linalg.norm(primal_adj-adj_matrix)
	signal_error_reference.extend([signal_error_ref])
	signal_error_list.extend([signal_error])
	graph_error_list.extend([graph_error])

	tr1=np.trace(np.dot(signals, np.dot(laplacian, signals.T) ))
	tr2=np.trace(np.dot(signals, np.dot(knn_lap, signals.T)))
	trace1.extend([tr1])
	trace2.extend([tr2])

	tr3=np.trace(np.dot(X, np.dot(laplacian, X.T) ))
	tr4=np.trace(np.dot(X, np.dot(knn_lap, X.T)))
	trace3.extend([tr3])
	trace4.extend([tr4])

	tr5=np.trace(np.dot(X_noise, np.dot(knn_lap, X_noise.T)))
	trace5.extend([tr5])

 




plt.plot(signal_error_reference, label='X_noise/X')
plt.plot(signal_error_list, label='Signal/X')
plt.title('signal error', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(graph_error_list)
plt.title('graph error', fontsize=12)
plt.show()


#plt.plot(trace1, label='signal/lap')
plt.plot(trace2, label='signal/knn')
#plt.plot(trace3, label='X/lap')
plt.plot(trace4, label='X/knn')
plt.plot(trace5, label='X_noise/knn')
plt.legend(loc=1)
plt.show()


s=np.array(smoothness)
for i in range(len(s)):
	plt.plot(s[i], label='%s'%(i))
plt.legend(loc=1)
plt.show()


real_signal=X[0,:]
#noise_signal=X
learned_signal=signals[0,:]

real_graph=create_networkx_graph(node_num, adj_matrix)
edge_num=real_graph.number_of_edges()
edge_weights=adj_matrix[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(real_graph, knn_pos, node_color=real_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(real_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()



#primal_adj=filter_graph_to_knn(primal_adj, node_num)
primal_adj[primal_adj<10**(-4)]=0
learned_graph=create_networkx_graph(node_num, primal_adj)
edge_num=learned_graph.number_of_edges()
edge_weights=primal_adj[np.triu_indices(node_num,0)]
edge_color=edge_weights[edge_weights>0]
edge_alpha=edge_color
nodes=nx.draw_networkx_nodes(learned_graph, knn_pos, node_color=learned_signal,node_size=100, cmap=plt.cm.Reds)
edges=nx.draw_networkx_edges(learned_graph, knn_pos, width=1.0, alpha=1.0, edge_color=edge_color, edge_cmap=plt.cm.Blues,)
plt.axis('off')
plt.show()

plt.plot(error)
plt.title('graph learning error', fontsize=12)
plt.show()


fig,(ax1, ax2)=plt.subplots(1,2, figsize=(4,2))
ax1.pcolor(adj_matrix, cmap='RdBu')
ax1.set_title('real W')
ax2.pcolor(primal_adj, cmap='RdBu')
ax2.set_title('learned w')
plt.show()
