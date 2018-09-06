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
import seaborn as sns
sns.set_style('white')
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 


user_num=100
item_num=1000
dimension=20
item_pool_size=5
cluster_num=4
cluster_std=0.1
noise_scale=0.1
gl_alpha=1
gl_beta=0.2
gl_theta=0.01
gl_step_size=0.5
alpha=0.05
iteration=5000

newpath=path+'LINUCB_user_num_%s_cluster_num_%s'%(user_num, cluster_num)+'/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)


user_pool=generate_all_random_users(iteration, user_num)
item_pools=generate_all_article_pool(iteration, item_pool_size, item_num)


combination_list=[]
regret_dict={}
learning_error_dict={}


noisy_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)
true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj,0)

plt.scatter(true_user_features[:,0], true_user_features[:,1], c=true_label, cmap=plt.cm.jet)
plt.savefig(newpath+'graph_noise_cluster_%s_%s'%(noise_scale, cluster_std)+'.png', dpi=100)
plt.clf()

linucb_mab=LINUCB_MAB(user_num, item_num, dimension, item_pool_size, alpha, true_user_features=true_user_features, true_graph=None)
linucb_cum_regret,linucb_user_f, linucb_error=linucb_mab.run(user_pool, item_pools, item_features, noisy_signal, iteration)

key=str(noise_scale)+str(cluster_std)
combination_list.extend([key])
regret_dict[key]=linucb_cum_regret
learning_error_dict[key]=linucb_error

np.save(newpath+'combination_list'+'.npy', combination_list)
np.save(newpath+'regret_dict'+'.npy', regret_dict)
np.save(newpath+'learning_error_dict'+'.npy', learning_error_dict)


plt.plot(linucb_cum_regret, label='LinUCB')
plt.ylabel('Cum Regret', fontsize=12)
plt.legend(loc=1)
plt.show()

plt.plot(linucb_error, label='LinUCB')
plt.ylabel('Learning Error', fontsize=12)
plt.legend(loc=1)
plt.show()








