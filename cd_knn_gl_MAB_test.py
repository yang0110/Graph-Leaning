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
from sklearn.metrics.pairwise import rbf_kernel

user_num=20
item_num=1000
dimension=2
item_pool_size=25
cluster_num=4
cluster_std=1.0
noise_scale=0.5
gl_alpha=0.1
gl_beta=0.1
gl_theta=0.1
gl_step_size=0.5
alpha=0.05
iteration=200

noisy_signal, item_features, true_user_features=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)

true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj,0)
user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)
#########################

knn_mab=KNN_MAB(user_num, item_num, dimension, item_pool_size,alpha, K=10, true_user_features=true_user_features, true_graph=None)
knn_cum_regret, knn_adj,knn_user_f, knn_error, knn_denoised_signal=knn_mab.run(user_pool, item_pools, item_features, noisy_signal, iteration)

gl_mab=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, true_user_features=true_user_features)

gl_cum_regret, gl_adj, gl_user_f, gl_error, gl_denoised_signal=gl_mab.run(user_pool, item_pools, item_features, noisy_signal, iteration)

plt.plot(gl_cum_regret, label='GL')
plt.plot(knn_cum_regret, label='KNN')
plt.ylabel('Cum Regret', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(gl_errorm, label='GL')
plt.plot(knn_error, label='KNN')
plt.ylabel('Learning Error (User Feature)', fontsize=12)
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

pos=true_user_features
graph=create_networkx_graph(user_num, knn_adj)
edge_color=knn_adj[np.triu_indices(user_num,1)]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=knn_denoised_signal[0], node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color='grey')
plt.axis('off')
plt.title('KNN Graph', fontsize=12)
plt.show()