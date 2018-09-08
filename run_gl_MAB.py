import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx 
import os
os.chdir('D:/Research/Graph Learning/code/')
from collections import Counter
import datetime
from synthetic_data import *
from gl_MAB import GL_MAB
from knn_MAB import KNN_MAB
from cd_MAB import CD_MAB
from linucb_MAB import LINUCB_MAB
from sklearn.metrics.pairwise import rbf_kernel
path='D:/Research/Graph Learning/code/results/MAB_models_results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

user_num=20
item_num=1000
dimension=5
item_pool_size=5
cluster_num=4
cluster_std=0.1
noise_scale=0.1
gl_alpha=1
gl_beta=0.2
gl_theta=0.01
gl_step_size=0.5
jump_step=10
alpha=0.05
iteration=2000

newpath=path+'GL_user_num_%s_cluster_num_%s'%(user_num, cluster_num)+'/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)

combination_list=[]
regret_dict={}
learning_error_dict={}
graph_error_dict={}
denoised_signal_dict={}
adj_dict={}


noisy_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)
true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj,0)

gl_mab_1=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=1,  true_user_features=true_user_features, true_graph=true_adj)
gl_cum_regret_1, gl_adj_1, gl_user_f_1, gl_error_1, gl_graph_error_1, gl_denoised_signal_1=gl_mab_1.run(user_pool, item_pools, item_features, noisy_signal, iteration)


gl_mab_2=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=2,  true_user_features=true_user_features, true_graph=true_adj)
gl_cum_regret_2, gl_adj_2, gl_user_f_2, gl_error_2, gl_graph_error_2, gl_denoised_signal_2=gl_mab_2.run(user_pool, item_pools, item_features, noisy_signal, iteration)


gl_mab_3=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=3,  true_user_features=true_user_features, true_graph=true_adj)
gl_cum_regret_3, gl_adj_3, gl_user_f_3, gl_error_3, gl_graph_error_3, gl_denoised_signal_3=gl_mab_3.run(user_pool, item_pools, item_features, noisy_signal, iteration)

plt.plot(gl_cum_regret_1, label='Mode 1')
plt.plot(gl_cum_regret_2, label='Mode 2')
plt.plot(gl_cum_regret_3, label='Mode 3')
plt.legend(loc=2)
plt.title('regret')
plt.show()


plt.plot(gl_error_1, label='Mode 1')
plt.plot(gl_error_2, label='Mode 2')
plt.plot(gl_error_3, label='Mode 3')
plt.legend(loc=2)
plt.title('learning error')
plt.show()


plt.plot(gl_graph_error_1, label='Mode 1')
plt.plot(gl_graph_error_2, label='Mode 2')
plt.plot(gl_graph_error_3, label='Mode 3')
plt.legend(loc=2)
plt.title('graph error')
plt.show()



## save data
key=noise_scale
combination_list.extend([key])
regret_dict[key]=gl_cum_regret
learning_error_dict[key]=gl_error
denoised_signal_dict[key]=gl_denoised_signal
graph_error_dict[key]=gl_graph_error
adj_dict[key]=gl_adj

np.save(newpath+'combination_list'+'.npy', combination_list)
np.save(newpath+'regret_dict'+'.npy', regret_dict)
np.save(newpath+'learning_error_dict'+'.npy', learning_error_dict)
np.save(newpath+'denoised_signal_dict'+'.npy', denoised_signal_dict)
np.save(newpath+'adj_dict'+'.npy', adj_dict)
np.save(newpath+'graph_error_dict'+'.npy', graph_error_dict)



plt.figure()
plt.plot(gl_cum_regret, label='GL')
plt.ylabel('Cum Regret', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.figure()
plt.plot(gl_error, label='GL')
plt.ylabel('Learning Error', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.figure()
plt.plot(gl_graph_error, label='GL')
plt.ylabel('graph Error', fontsize=12)
plt.legend(loc=1)
plt.show()




pos=true_user_features
fig, (ax1, ax2)=plt.subplots(1,2, figsize=(6,3))
ax1.scatter(pos[:,0], pos[:,1], c=noisy_signal[0].tolist(), cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=gl_denoised_signal[0].tolist(), cmap=plt.cm.jet)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Noisy Signal')
ax2.set_title('GL Signal')
plt.tight_layout()
plt.show()

test_item=np.random.normal(size=dimension)
true_payoff=np.dot(true_user_features, test_item)
gl_payoff=np.dot(gl_user_f, test_item)

pos=true_user_features
graph=create_networkx_graph(user_num, true_adj)
edge_color=true_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=true_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.show()


pos=true_user_features
graph=create_networkx_graph(user_num, gl_adj)
edge_color=gl_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=gl_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('GL Graph', fontsize=12)
plt.show()
