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
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

user_num=50
item_num=1000
dimension=20
item_pool_size=25
cluster_num=4
cluster_std=0.1
noise_scale=0.1
gl_alpha=1
gl_beta=0.2
gl_theta=0.1
gl_step_size=0.5
alpha=0.05
iteration=2000

newpath=path+'GL_user_num_%s_cluster_num_%s'%(user_num, cluster_num)+'/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)

noise_scale_list=[0.1, 0.25, 0.5]
cluster_std_list=[0.1, 0.25, 0.5]
combination_list=[]
regret_dict={}
learning_error_dict={}
denoised_signal_dict={}
adj_dict={}
for noise_scale in noise_scale_list:
	for cluster_std in cluster_std_list:
		#create data
		print('noise_scale, cluster_s', noise_scale, cluster_std)
		noisy_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)
		true_adj=rbf_kernel(true_user_features)
		np.fill_diagonal(true_adj,0)
		##run model
		gl_mab=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, true_user_features=true_user_features)
		gl_cum_regret, gl_adj, gl_user_f, gl_error, gl_denoised_signal=gl_mab.run(user_pool, item_pools, item_features, noisy_signal, iteration)

		## save data
		key=str(noise_scale)+str(cluster_std)
		combination_list.extend([key])
		regret_dict[key]=gl_cum_regret
		learning_error_dict[key]=gl_error
		denoised_signal_dict[key]=gl_denoised_signal
		adj_dict[key]=gl_adj

np.save(newpath+'combination_list'+'.npy', combination_list)
np.save(newpath+'regret_dict'+'.npy', regret_dict)
np.save(newpath+'learning_error_dict'+'.npy', learning_error_dict)
np.save(newpath+'denoised_signal_dict'+'.npy', denoised_signal_dict)
np.save(newpath+'adj_dict'+'.npy', adj_dict)



plt.plot(gl_cum_regret, label='GL')
plt.ylabel('Cum Regret', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(gl_error, label='GL')
plt.ylabel('Learning Error', fontsize=12)
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
graph=create_networkx_graph(user_num, gl_adj)
edge_color=gl_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=gl_denoised_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('GL Graph', fontsize=12)
plt.show()




fig, (ax1, ax2)=plt.subplots(1,2, figsize=(6,3))
ax1.scatter(pos[:,0], pos[:,1], c=noisy_signal[0].tolist(), cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=gl_denoised_signal[0].tolist(), cmap=plt.cm.jet)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Noisy Signal')
ax2.set_title('GL Signal')

plt.tight_layout()
plt.show()
