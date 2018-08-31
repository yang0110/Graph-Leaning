import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import seaborn as sns
sns.set_style("white")
from synthetic_data import *
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from gl_algorithms import *
path='C:/Kaige_Research/Graph Learning/graph_learning_code/results/'
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 

node_num=100
signal_num=100
error_sigma=0.01
## graphs
rgg_adj, rgg_lap, rgg_pos=RGG(node_num)
rbf_adj, rbf_lap, rbf_pos=rbf_graph(node_num)
knn_adj, knn_lap, knn_pos=knn_graph(node_num)
er_adj, er_lap=er_graph(node_num)
ba_adj, ba_lap=ba_graph(node_num)
er_pos=rgg_pos
ba_pos=rgg_pos

## linear signals
rgg_signal=generate_signal(signal_num, node_num, rgg_pos)
rbf_signal=generate_signal(signal_num, node_num, rbf_pos)
knn_signal=generate_signal(signal_num, node_num, knn_pos)
er_signal=generate_signal(signal_num, node_num, er_pos)
ba_signal=generate_signal(signal_num, node_num, ba_pos)
### gl_siprep signal
rgg_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rgg_lap, error_sigma)
rbf_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap, error_sigma)
knn_gl_signal=generate_signal_gl_siprep(signal_num, node_num, knn_lap, error_sigma)
er_gl_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap, error_sigma)
ba_gl_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap, error_sigma)

## nonlinear signal: f1, f2, f3, f4, f5
rgg_f=np.array([f1(rgg_pos[:,0], rgg_pos[:,1]), f2(rgg_pos[:,0], rgg_pos[:,1]), f3(rgg_pos[:,0], rgg_pos[:,1]), f4(rgg_pos[:,0], rgg_pos[:,1]), f5(rgg_pos[:,0], rgg_pos[:,1])])
rbf_f=np.array([f1(rbf_pos[:,0], rbf_pos[:,1]), f2(rbf_pos[:,0], rbf_pos[:,1]), f3(rbf_pos[:,0], rbf_pos[:,1]), f4(rbf_pos[:,0], rbf_pos[:,1]), f5(rbf_pos[:,0], rbf_pos[:,1])])
knn_f=np.array([f1(knn_pos[:,0], knn_pos[:,1]), f2(knn_pos[:,0], knn_pos[:,1]), f3(knn_pos[:,0], knn_pos[:,1]), f4(knn_pos[:,0], knn_pos[:,1]), f5(knn_pos[:,0], knn_pos[:,1])])
er_f=np.array([f1(er_pos[:,0], er_pos[:,1]), f2(er_pos[:,0], er_pos[:,1]), f3(er_pos[:,0], er_pos[:,1]), f4(er_pos[:,0], er_pos[:,1]), f5(er_pos[:,0], er_pos[:,1])])
ba_f=np.array([f1(ba_pos[:,0], ba_pos[:,1]), f2(ba_pos[:,0], ba_pos[:,1]), f3(ba_pos[:,0], ba_pos[:,1]), f4(ba_pos[:,0], ba_pos[:,1]), f5(ba_pos[:,0], ba_pos[:,1])])

## plot
rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
plot_graph_and_signal(rgg_adj_filtered, rgg_signal[0], rgg_pos, node_num, error_sigma)
plot_graph_and_signal(rbf_adj, rbf_signal[0], rbf_pos, node_num, error_sigma)
plot_graph_and_signal(knn_adj, knn_signal[0], knn_pos, node_num, error_sigma)
plot_graph_and_signal(er_adj, er_signal[0], er_pos, node_num, error_sigma)
plot_graph_and_signal(ba_adj, ba_signal[0], ba_pos, node_num, error_sigma)

###
rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
plot_graph_and_signal(rgg_adj_filtered, rgg_gl_signal[0], rgg_pos, node_num, error_sigma)
plot_graph_and_signal(rbf_adj, rbf_gl_signal[0], rbf_pos, node_num, error_sigma)
plot_graph_and_signal(knn_adj, knn_gl_signal[0], knn_pos, node_num, error_sigma)
plot_graph_and_signal(er_adj, er_gl_signal[0], er_pos, node_num, error_sigma)
plot_graph_and_signal(ba_adj, ba_gl_signal[0], ba_pos, node_num, error_sigma)


###
rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
plot_graph_and_signal(rgg_adj_filtered, rgg_f[0], rgg_pos, node_num, error_sigma)
plot_graph_and_signal(rbf_adj, rbf_f[0], rbf_pos, node_num, error_sigma)
plot_graph_and_signal(knn_adj, knn_f[0], knn_pos, node_num, error_sigma)
plot_graph_and_signal(er_adj, er_f[0], er_pos, node_num, error_sigma)
plot_graph_and_signal(ba_adj, ba_f[0], ba_pos, node_num, error_sigma)


smooth1_1=calculate_smoothness(rgg_gl_signal, rgg_lap)
smooth1_2=calculate_smoothness(rbf_gl_signal, rbf_lap)
smooth1_3=calculate_smoothness(knn_gl_signal, knn_lap)
smooth1_4=calculate_smoothness(er_gl_signal, er_lap)
smooth1_5=calculate_smoothness(ba_gl_signal, ba_lap)

smooth2_1=calculate_smoothness(rgg_signal, rgg_lap)
smooth2_2=calculate_smoothness(rbf_signal, rbf_lap)
smooth2_3=calculate_smoothness(knn_signal, knn_lap)
smooth2_4=calculate_smoothness(er_signal, er_lap)
smooth2_5=calculate_smoothness(ba_signal, ba_lap)

smooth3_1=calculate_smoothness(rgg_f, rgg_lap)
smooth3_2=calculate_smoothness(rbf_f, rbf_lap)
smooth3_3=calculate_smoothness(knn_f, knn_lap)
smooth3_4=calculate_smoothness(er_f, er_lap)
smooth3_5=calculate_smoothness(ba_f, ba_lap)

fig, axe=plt.subplots(2,2)
axe[0,0].plot(smooth1_1, label='1_1')
axe[0,0].plot(smooth1_2, label='1_2')
axe[0,0].plot(smooth1_3, label='1_3')
axe[0,0].plot(smooth1_4, label='1_4')
axe[0,0].plot(smooth1_5, label='1_5')
axe[0,0].set_title('Factor model')
axe[1,0].plot(smooth2_1, label='2_1')
axe[1,0].plot(smooth2_2, label='2_2')
axe[1,0].plot(smooth2_3, label='2_3')
axe[1,0].plot(smooth2_4, label='2_4')
axe[1,0].plot(smooth2_5, label='2_5')
axe[1,0].set_title('Linear model')
axe[1,1].plot(smooth3_1, label='3_1')
axe[1,1].plot(smooth3_2, label='3_2')
axe[1,1].plot(smooth3_3, label='3_3')
axe[1,1].plot(smooth3_4, label='3_4')
axe[1,1].plot(smooth3_5, label='3_5')
axe[1,1].set_title('non linear')
plt.legend(loc=1)
plt.show()