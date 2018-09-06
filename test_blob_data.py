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

user_num=50
item_num=1000
dimension=20
cluster_num=4
cluster_std=1.0
noise_scale=0.1

noisy_signal, item_features, true_user_features, true_label=blob_data(user_num, item_num, dimension, cluster_num, cluster_std, noise_scale)
true_adj=rbf_kernel(true_user_features)
np.fill_diagonal(true_adj,0)

plt.scatter(true_user_features[:,0], true_user_features[:,1], c=noisy_signal[0], cmap=plt.cm.jet)
plt.title('Cluster Std %s'%(cluster_std), fontsize=12)
plt.show()


