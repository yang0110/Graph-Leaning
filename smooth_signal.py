import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('D:/Research/Graph Learning/code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import seaborn as sns
sns.set_style("white")
from synthetic_data import *
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from primal_dual_gl import Primal_dual_gl 
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd

path='D:/Research/Graph Learning/code/results/MAB_models_results/'

node_num=20
signal_num=100
error_sigma=0.01
dimension=2
threshold=0.75 
target_trace=node_num

pos=np.random.uniform(size=(node_num, dimension))
rbf_adj, rbf_lap=generate_rbf_graph(node_num, pos, threshold=threshold)
er_adj, er_lap=generate_er_graph(node_num)
ba_adj, ba_lap=generate_ba_graph(node_num)

rbf_noisy_signal, rbf_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap, error_sigma)
er_noisy_signal, er_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap, error_sigma)
ba_noisy_signal, ba_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap, error_sigma)

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=rbf_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=rbf_noisy_signal[0], cmap=plt.cm.jet)
plt.show()

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=er_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=er_noisy_signal[0], cmap=plt.cm.jet)
plt.show()

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.scatter(pos[:,0], pos[:,1], c=ba_signal[0], cmap=plt.cm.jet)
ax2.scatter(pos[:,0], pos[:,1], c=ba_noisy_signal[0], cmap=plt.cm.jet)
plt.show()



noisy_signal=ba_noisy_signal
true_lap=ba_lap

Z=euclidean_distances(noisy_signal.T, squared=True)
np.fill_diagonal(Z,0)
Z=norm_W(Z, node_num)
primal_gl=Primal_dual_gl(node_num, Z, alpha=1, beta=0.1, step_size=0.5)
primal_adj, error=primal_gl.run()
primal_lap=csgraph.laplacian(primal_adj, normed=False)
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.pcolor(true_lap, cmap=plt.cm.jet)
ax2.pcolor(primal_lap, cmap=plt.cm.jet)
plt.show()

U, Sigma, VT = randomized_svd(noisy_signal, 
                              n_components=20,
                              n_iter=5,
                              random_state=None)
item_f=U
sigma=np.diag(Sigma)
user_f=np.dot(sigma, VT)
a=np.dot(item_f, user_f)