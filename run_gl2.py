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
dimension=10
item_pool_size=25
cluster_num=5
cluster_std=0.1
noise_scale=0.1
K=10
gl_alpha=1
gl_beta=0.1
gl_theta=0.01
gl_step_size=0.5
jump_step=50
alpha=0.05
iteration=5000


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

gl_theta_list=[0.01, 0.05, 0.1, 0.2, 0.3]
regret_dict={}
signal_error_dict={}
graph_error_dict={}
for gl_theta in gl_theta_list:
	gl2_mab=GL_MAB2(user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size, jump_step=jump_step, mode=1, true_user_features=true_user_features, true_graph=true_adj)
	gl2_cum_regret, gl2_adj, gl2_user_f, gl2_error, gl2_graph_error, gl2_denoised_signal=gl2_mab.run(user_pool, item_pools, item_features, noisy_signal,true_signal,  iteration)
	regret_dict[gl_theta]=gl2_cum_regret
	signal_error_dict[gl_theta]=gl2_error
	graph_error_dict[gl_theta]=gl2_graph_error






plt.figure(figsize=(5,5))
for theta in gl_theta_list:
	plt.plot(regret_dict[theta], label=theta)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=2)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.savefig(newpath+'cum_regret_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.tight_layout()
plt.clf()


plt.figure(figsize=(5,5))
for theta in gl_theta_list:
	plt.plot(signal_error_dict[tehta], label=theta)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.legend(loc=1)
plt.tight_layout()
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.savefig(newpath+'learning_error_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


plt.figure(figsize=(5,5))
for theta in gl_theta_list:
	plt.plot(graph_error_dict[theta], label=theta)
plt.ylabel('Graph Error', fontsize=12)
plt.xlabel('Time', fontsize=12)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig(newpath+'graph_error_%s_%s_%s'%(user_num,int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
plt.clf()


# pos=true_user_features[:,:2]
# test_item=np.random.normal(size=dimension)
# true_payoff=np.dot(true_user_features, test_item)
# noise=np.random.normal(size=user_num)
# noisy_payoff=true_payoff+noise
# gl_payoff=np.dot(gl2_user_f, test_item)

# plt.figure(figsize=(5,5))
# plt.scatter(pos[:,0], pos[:,1], c=true_payoff, cmap=plt.cm.jet)
# plt.title('True signal', fontsize=12)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.tight_layout()
# plt.savefig(newpath+'true_signal_%s_%s_%s_cluster_std_%s'%(user_num, dimension, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()


# plt.figure(figsize=(5,5))
# plt.scatter(pos[:,0], pos[:,1], c=noisy_payoff, cmap=plt.cm.jet)
# plt.title('Noisy signal', fontsize=12)
# plt.xlabel('x', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.tight_layout()
# plt.savefig(newpath+'noisy_signal_%s_%s_%s_cluster_std_%s'%(user_num, dimension, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()

# fig, axes=plt.subplots(3,2)
# axes[0,0].scatter(pos[:,0], pos[:,1], c=true_payoff, cmap=plt.cm.jet)
# axes[0,1].scatter(pos[:,0], pos[:,1], c=noisy_payoff, cmap=plt.cm.jet)
# axes[2,1].scatter(pos[:,0], pos[:,1], c=gl_payoff, cmap=plt.cm.jet)
# axes[0,0].set_title('True signal')
# axes[0,1].set_title('Noisy Signal')
# axes[2,1].set_title('GL Signal')
# plt.tight_layout()
# plt.savefig(newpath+'signal_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()


# fig, axes=plt.subplots(2,2)
# axes[0,0].pcolor(true_adj, cmap=plt.cm.jet)
# axes[1,1].pcolor(gl2_adj, cmap=plt.cm.jet)
# axes[0,0].set_title('True Adj')
# axes[1,1].set_title('GL Adj')
# plt.tight_layout()
# plt.savefig(newpath+'Adj_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()




# gl_payoff2=total_variation_signal_learning(gl2_adj, noisy_payoff)	
# fig, (ax1, ax2)=plt.subplots(2)
# ax1.scatter(pos[:,0], pos[:,1], c=gl_payoff, cmap=plt.cm.jet)
# ax1.set_title('GL Tikhonov Reg')
# ax2.scatter(pos[:,0], pos[:,1], c=gl_payoff2, cmap=plt.cm.jet)
# ax2.set_title('GL TV Reg')
# plt.tight_layout()
# plt.savefig(newpath+'Tik_VS_TV_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()



# graph=create_networkx_graph(user_num, true_adj)
# edge_weight=true_adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_color=true_payoff, node_size=100, cmap=plt.cm.jet)
# edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('True Graph', fontsize=12)
# plt.savefig(newpath+'True_Graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()


# graph=create_networkx_graph(user_num, true_adj)
# edge_weight=true_adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_color=noisy_payoff, node_size=100, cmap=plt.cm.jet)
# edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('True Graph', fontsize=12)
# plt.savefig(newpath+'noisy_Graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()


# graph=create_networkx_graph(user_num, gl2_adj)
# edge_weight=gl2_adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_color=gl_payoff, node_size=100, cmap=plt.cm.jet)
# edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.gray)
# plt.axis('off')
# plt.title('GL Graph', fontsize=12)
# plt.savefig(newpath+'GL_graph_%s_%s_%s'%(user_num, int(100*noise_scale), int(100*cluster_std))+'.png', dpi=100)
# plt.clf()



