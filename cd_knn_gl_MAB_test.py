import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import networkx as nx 
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from collections import Counter
import datetime
from synthetic_data import *
from utils import *
from gl_MAB import GL_MAB
from gl_MAB2 import GL_MAB2
from knn_MAB import KNN_MAB
from correct_knn_MAB import Correct_KNN_MAB
from cd_MAB import CD_MAB
from linucb_MAB import LINUCB_MAB
from sklearn.metrics.pairwise import rbf_kernel
from pygsp import graphs, plotting, filters
import pyunlocbox
import seaborn as sns 
from sklearn.utils.extmath import randomized_svd
sns.set_style('white')

path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/MAB_models_results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 


user_num=20
item_num=1000
dimension=2
item_pool_size=25
cluster_num=5
cluster_std=0.1
noise_scale=0.25
K=10
gl_alpha=1
gl_beta=0.2
gl_theta=0.01
gl_step_size=0.5
jump_step=10
alpha=0.05
iteration=1000


newpath=path+'user_num_%s_dimension_%s_noise_scale_%s_cluster_std_%s'%(user_num, dimension, noise_scale, cluster_std)+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

noisy_signal, true_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)

plt.figure(figsize=(5,5))
plt.plot(noisy_signal[0], label='noisy')
plt.plot(true_signal[0], label='true')
plt.legend(loc=2)
plt.ylabel('signal', fontsize=12)
plt.xlabel('node index', fontsize=12)
plt.savefig(newpath+'true_vs_noisy_signal_%s'%(noise_scale)+'.png', dpi=100)
plt.clf()

true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj, 0)
true_adj=norm_W(true_adj, user_num)
user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)
#########################

linucb_mab=LINUCB_MAB(user_num, item_num, dimension, item_pool_size, alpha, true_user_features=true_user_features, true_graph=true_adj)
linucb_cum_regret, linucb_user_f, linucb_error=linucb_mab.run(user_pool, item_pools, item_features, noisy_signal, true_signal, iteration)

cd_mab=CD_MAB(user_num, item_num, dimension, item_pool_size, alpha, K=5, jump_step=10, true_user_features=true_user_features, true_graph=true_adj)
cd_cum_regret, cd_adj, cd_user_f, cd_cluster_f, cd_error, cd_graph_error, cd_cluster=cd_mab.run(user_pool, item_pools, item_features, noisy_signal,true_signal, iteration)

knn_mab=KNN_MAB(user_num, item_num, dimension, item_pool_size,alpha, K=user_num, jump_step=10, true_user_features=true_user_features, true_graph=true_adj)
knn_cum_regret, knn_adj, knn_user_f, knn_error, knn_graph_error,  knn_denoised_signal=knn_mab.run(user_pool, item_pools, item_features, noisy_signal, true_signal, iteration)

c_knn_mab=Correct_KNN_MAB(user_num, item_num, dimension, item_pool_size,alpha, K=user_num, jump_step=10, true_user_features=true_user_features, true_graph=true_adj)
c_knn_cum_regret, c_knn_adj,c_knn_user_f, c_knn_error, c_knn_graph_error,  c_knn_denoised_signal, c_knn_s_list=c_knn_mab.run(user_pool, item_pools, item_features, noisy_signal, true_signal, iteration)

gl_mab=GL_MAB(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=1, true_user_features=true_user_features, true_graph=true_adj)
gl_cum_regret, gl_adj, gl_user_f, gl_error, gl_graph_error, gl_denoised_signal=gl_mab.run(user_pool, item_pools, item_features, noisy_signal,true_signal,  iteration)

gl2_mab=GL_MAB2(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=1, true_user_features=true_user_features, true_graph=true_adj)
gl2_cum_regret, gl2_adj, gl2_user_f, gl2_error, gl2_graph_error, gl2_denoised_signal=gl2_mab.run(user_pool, item_pools, item_features, noisy_signal,true_signal,  iteration)


plt.plot(c_knn_s_list)
plt.ylabel('smoothness')
plt.clf()

plt.figure(figsize=(5,5))
plt.plot(gl2_cum_regret, label='GL', color='r', marker='o', markersize=5, markevery=0.1)
plt.plot(gl_cum_regret, label='GL+KNN', color='k', marker='o', markersize=5, markevery=0.1)
plt.plot(knn_cum_regret, label='RBF+KNN', color='y', marker='*',markersize=5,  markevery=0.1)
plt.plot(c_knn_cum_regret, label='KNN Correct', color='c', marker='*', markersize=8, markevery=0.1)
plt.plot(cd_cum_regret, label='CD', color='b', marker='p',markersize=5,  markevery=0.1)
plt.plot(linucb_cum_regret, label='LINUCB', color='g', marker='s',markersize=5,  markevery=0.1)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.savefig(newpath+'cum_regret_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.tight_layout()
plt.clf()


plt.figure(figsize=(5,5))
plt.plot(gl2_error, label='GL', color='r', marker='o', markersize=5, markevery=0.1)
plt.plot(gl_error, label='GL+KNN', color='k', marker='o', markersize=5, markevery=0.1)
plt.plot(knn_error, label='RBF+KNN', color='y', marker='*',markersize=5,  markevery=0.1)
plt.plot(c_knn_error, label='KNN Correct', color='c', marker='*', markersize=5, markevery=0.1)
plt.plot(cd_error, label='CD', color='b', marker='p',markersize=5,  markevery=0.1)
plt.plot(linucb_error, label='LINUCB', color='g', marker='s',markersize=5,  markevery=0.1)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1)
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.savefig(newpath+'learning_error_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()




plt.figure(figsize=(5,5))
plt.plot(gl2_graph_error, label='GL', color='r', marker='o', markersize=5, markevery=0.3)
plt.plot(gl_graph_error, label='GL+KNN', color='k', marker='o', markersize=5, markevery=0.3)
plt.plot(knn_graph_error, label='RBF+KNN', color='y', marker='*',markersize=5,  markevery=0.3)
plt.plot(c_knn_graph_error, label='KNN Correct', color='c', marker='*', markersize=5, markevery=0.1)
plt.plot(cd_graph_error, label='CD', color='b', marker='p',markersize=5,  markevery=0.3)
plt.ylabel('Graph Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig(newpath+'graph_error_%s_%s_%s'%(user_num,int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


plt.figure(figsize=(5,5))
plt.plot(linucb_cum_regret, label='LINUCB',color='g', marker='s',markersize=5,  markevery=0.1)
plt.plot(gl2_cum_regret, label='GL',color='r', marker='o', markersize=5, markevery=0.1)
plt.legend(loc=1)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'linucb_VS_GL_cum_regret_%s_%s_%s'%(user_num,int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()

plt.figure(figsize=(5,5))
plt.plot(linucb_error, label='LINUCB',color='g', marker='s',markersize=5,  markevery=0.1)
plt.plot(gl2_error, label='GL',color='r', marker='o', markersize=5, markevery=0.1)
plt.legend(loc=1)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'linucb_VS_GL_learning_error_%s_%s_%s'%(user_num,int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()

pos=true_user_features[:,:2]
test_item=np.random.normal(size=dimension)
true_payoff=np.dot(true_user_features, test_item)
noise=np.random.normal(size=user_num)
noisy_payoff=true_payoff+noise
linucb_payoff=np.dot(linucb_user_f, test_item)
cd_payoff=np.dot(cd_user_f, test_item)
knn_payoff=np.dot(knn_user_f, test_item)
gl_payoff=np.dot(gl2_user_f, test_item)

plt.figure(figsize=(5,5))
plt.scatter(pos[:,0], pos[:,1], c=true_payoff, cmap=plt.cm.jet)
plt.title('True signal', fontsize=12)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'true_signal_%s_%s_%s_cluster_std_%s'%(user_num, dimension, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


plt.figure(figsize=(5,5))
plt.scatter(pos[:,0], pos[:,1], c=noisy_payoff, cmap=plt.cm.jet)
plt.title('Noisy signal', fontsize=12)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.tight_layout()
plt.savefig(newpath+'noisy_signal_%s_%s_%s_cluster_std_%s'%(user_num, dimension, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()

fig, axes=plt.subplots(3,2)
axes[0,0].scatter(pos[:,0], pos[:,1], c=true_payoff, cmap=plt.cm.jet)
axes[0,1].scatter(pos[:,0], pos[:,1], c=noisy_payoff, cmap=plt.cm.jet)
axes[1,0].scatter(pos[:,0], pos[:,1], c=linucb_payoff, cmap=plt.cm.jet)
axes[1,1].scatter(pos[:,0], pos[:,1], c=cd_payoff, cmap=plt.cm.jet)
axes[2,0].scatter(pos[:,0], pos[:,1], c=knn_payoff, cmap=plt.cm.jet)
axes[2,1].scatter(pos[:,0], pos[:,1], c=gl_payoff, cmap=plt.cm.jet)
axes[0,0].set_title('True signal')
axes[0,1].set_title('Noisy Signal')
axes[1,0].set_title('LINUCB Signal')
axes[1,1].set_title('CD Signal')
axes[2,0].set_title('KNN Signal')
axes[2,1].set_title('GL Signal')
plt.tight_layout()
plt.savefig(newpath+'signal_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


fig, axes=plt.subplots(2,2)
axes[0,0].pcolor(true_adj, cmap=plt.cm.jet)
axes[0,1].pcolor(cd_adj, cmap=plt.cm.jet)
axes[1,0].pcolor(knn_adj, cmap=plt.cm.jet)
axes[1,1].pcolor(gl_adj, cmap=plt.cm.jet)
axes[0,0].set_title('True Adj')
axes[0,1].set_title('CD Adj')
axes[1,0].set_title('RBF Adj')
axes[1,1].set_title('GL Adj')
plt.tight_layout()
plt.savefig(newpath+'Adj_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()




gl_payoff2=total_variation_signal_learning(gl2_adj, noisy_payoff)	
fig, (ax1, ax2)=plt.subplots(2)
ax1.scatter(pos[:,0], pos[:,1], c=gl_payoff, cmap=plt.cm.jet)
ax1.set_title('GL Tikhonov Reg')
ax2.scatter(pos[:,0], pos[:,1], c=gl_payoff2, cmap=plt.cm.jet)
ax2.set_title('GL TV Reg')
plt.tight_layout()
plt.savefig(newpath+'Tik_VS_TV_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()



graph=create_networkx_graph(user_num, true_adj)
edge_weight=true_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=true_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.savefig(newpath+'True_Graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


graph=create_networkx_graph(user_num, true_adj)
edge_weight=true_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=noisy_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('True Graph', fontsize=12)
plt.savefig(newpath+'noisy_Graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


graph=create_networkx_graph(user_num, gl2_adj)
edge_weight=gl2_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=gl_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('GL Graph', fontsize=12)
plt.savefig(newpath+'GL_graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


graph=create_networkx_graph(user_num, knn_adj)
edge_weight=knn_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=knn_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('KNN Graph', fontsize=12)
plt.savefig(newpath+'KNN_graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


np.fill_diagonal(cd_adj,0)
graph=create_networkx_graph(user_num, cd_adj)
edge_weight=cd_adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_color=cd_payoff, node_size=100, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
plt.axis('off')
plt.title('CD Graph', fontsize=12)
plt.savefig(newpath+'CD_graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()