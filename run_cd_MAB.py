import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx 
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from collections import Counter
import datetime
from synthetic_data import *
from gl_MAB import GL_MAB
from knn_MAB import KNN_MAB
from cd_MAB import CD_MAB
from linucb_MAB import LINUCB_MAB
from sklearn.metrics.pairwise import rbf_kernel
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/MAB_models_results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 


user_num=100
item_num=1000
dimension=20
item_pool_size=5
cluster_num=4
cluster_std=0.1
noise_scale=0.1
K=10
gl_alpha=1
gl_beta=0.2
gl_theta=0.01
gl_step_size=0.5
alpha=0.05
iteration=5000

newpath=path+'CD_user_num_%s_cluster_num_%s'%(user_num, cluster_num)+'/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)


combination_list=[]
regret_dict={}
learning_error_dict={}
denoised_signal_dict={}
adj_dict={}


noisy_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)
true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj,0)

cd_mab=CD_MAB(user_num, item_num, dimension, item_pool_size, alpha, K=K, true_user_features=true_user_features, true_graph=true_adj)
cd_cum_regret, cd_adj, cd_user_f, cd_cluster_f, cd_error,cd_graph_error, cd_cluster, cd_cluster_score=cd_mab.run(user_pool, item_pools, item_features, noisy_signal, iteration, true_label)
key=str(noise_scale)+str(cluster_std)
combination_list.extend([key])
regret_dict[key]=cd_cum_regret
learning_error_dict[key]=cd_error
adj_dict[key]=cd_adj

np.save(newpath+'combination_list'+'.npy', combination_list)
np.save(newpath+'regret_dict'+'.npy', regret_dict)
np.save(newpath+'learning_error_dict'+'.npy', learning_error_dict)
np.save(newpath+'adj_dict'+'.npy', adj_dict)



plt.plot(cd_cum_regret, label='CD')
plt.ylabel('Cum Regret', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(cd_error, label='CD')
plt.ylabel('Learning Error', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(cd_graph_error, label='CD')
plt.ylabel('Graph Error', fontsize=12)
plt.legend(loc=1)
plt.show()


pos=true_user_features
graph=create_networkx_graph(user_num, true_adj)
edge_color=true_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=noisy_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.show()



pos=true_user_features
graph=create_networkx_graph(user_num,cd_adj)
edge_color=cd_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=cd_cluster, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('CD Graph', fontsize=12)
plt.show()



fig, (ax1, ax2)=plt.subplots(1,2, figsize=(6,3))
ax1.scatter(pos[:,0], pos[:,1], c=noisy_signal[0].tolist(), cmap=plt.cm.jet)
ax1.axis('off')
ax2.scatter(pos[:,0], pos[:,1], c=cd_cluster, cmap=plt.cm.jet)
ax2.axis('off')
ax1.set_title('Noisy Signal')
ax2.set_title('CD Cluster')
plt.tight_layout()
plt.show()
