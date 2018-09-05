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

gl_path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/GL_user_num_20_cluster_num_4/_09_05_18_15_38/'
knn_path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/KNN_user_num_20_cluster_num_4/_09_05_18_15_49/'
cd_path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/CD_user_num_50_cluster_num_4/_09_05_15_58_53/'
linucb_path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/LINUCB_user_num_50_cluster_num_4/_09_05_18_00_14/'

combination_list=np.load(cd_path+'combination_list.npy')
cd_cum_regret=np.load(cd_path+'regret_dict.npy').item()
cd_learning_error=np.load(cd_path+'learning_error_dict.npy').item()

linucb_cum_regret=np.load(linucb_path+'regret_dict.npy').item()
linucb_learning_error=np.load(linucb_path+'learning_error_dict.npy').item()

knn_cum_regret=np.load(knn_path+'regret_dict.npy').item()
knn_learning_error=np.load(knn_path+'learning_error_dict.npy').item()

gl_cum_regret=np.load(gl_path+'regret_dict.npy').item()
gl_learning_error=np.load(gl_path+'learning_error_dict.npy').item()
for i in combination_list:
	plt.figure()
	plt.plot(cd_cum_regret[i], label='CD')
	plt.plot(linucb_cum_regret[i], label='LINUCB')
	plt.plot(gl_cum_regret[i], label='GL')
	plt.plot(knn_cum_regret[i], label='KNN')
	plt.legend(loc=2)
	plt.title(i, fontsize=12)
	plt.ylabel('Regret')
	plt.show()


for i in combination_list:
	plt.figure()
	plt.plot(cd_learning_error[i], label='CD')
	plt.plot(linucb_learning_error[i], label='LINUCB')
	plt.plot(gl_learning_error[i], label='GL')
	plt.plot(knn_learning_error[i], label='KNN')
	plt.legend(loc=1)
	plt.title(i, fontsize=12)
	plt.ylabel('Learning Error')
	plt.show()