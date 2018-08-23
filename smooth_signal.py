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

node_num=20
signal_num=100

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
rgg_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rgg_lap)
rbf_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap)
knn_gl_signal=generate_signal_gl_siprep(signal_num, node_num, knn_lap)
er_gl_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap)
ba_gl_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap)
## nonlinear signal: f1, f2, f3, f4, f5
rgg_f=np.array([f1(rgg_pos[:,0], rgg_pos[:,1]), f2(rgg_pos[:,0], rgg_pos[:,1]), f3(rgg_pos[:,0], rgg_pos[:,1]), f4(rgg_pos[:,0], rgg_pos[:,1]), f5(rgg_pos[:,0], rgg_pos[:,1])])
rbf_f=np.array([f1(rbf_pos[:,0], rbf_pos[:,1]), f2(rbf_pos[:,0], rbf_pos[:,1]), f3(rbf_pos[:,0], rbf_pos[:,1]), f4(rbf_pos[:,0], rbf_pos[:,1]), f5(rbf_pos[:,0], rbf_pos[:,1])])
knn_f=np.array([f1(knn_pos[:,0], knn_pos[:,1]), f2(knn_pos[:,0], knn_pos[:,1]), f3(knn_pos[:,0], knn_pos[:,1]), f4(knn_pos[:,0], knn_pos[:,1]), f5(knn_pos[:,0], knn_pos[:,1])])
er_f=np.array([f1(er_pos[:,0], er_pos[:,1]), f2(er_pos[:,0], er_pos[:,1]), f3(er_pos[:,0], er_pos[:,1]), f4(er_pos[:,0], er_pos[:,1]), f5(er_pos[:,0], er_pos[:,1])])
ba_f=np.array([f1(ba_pos[:,0], ba_pos[:,1]), f2(ba_pos[:,0], ba_pos[:,1]), f3(ba_pos[:,0], ba_pos[:,1]), f4(ba_pos[:,0], ba_pos[:,1]), f5(ba_pos[:,0], ba_pos[:,1])])


### plot
# rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
# plot_graph_and_signal(rgg_adj_filtered, rgg_signal[0], rgg_pos, node_num)
# plot_graph_and_signal(rbf_adj, rbf_signal[0], rbf_pos, node_num)
# plot_graph_and_signal(knn_adj, knn_signal[0], knn_pos, node_num)
# plot_graph_and_signal(er_adj, er_signal[0], er_pos, node_num)
# plot_graph_and_signal(ba_adj, ba_signal[0], ba_pos, node_num)

# ###
# rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
# plot_graph_and_signal(rgg_adj_filtered, rgg_gl_signal[0], rgg_pos, node_num)
# plot_graph_and_signal(rbf_adj, rbf_gl_signal[0], rbf_pos, node_num)
# plot_graph_and_signal(knn_adj, knn_gl_signal[0], knn_pos, node_num)
# plot_graph_and_signal(er_adj, er_gl_signal[0], er_pos, node_num)
# plot_graph_and_signal(ba_adj, ba_gl_signal[0], ba_pos, node_num)

# ###
# rgg_adj_filtered=filter_graph_to_knn(rgg_adj, node_num, k=10)
# plot_graph_and_signal(rgg_adj_filtered, rgg_f, rgg_pos, node_num)
# plot_graph_and_signal(rbf_adj, rbf_f, rbf_pos, node_num)
# plot_graph_and_signal(knn_adj, knn_f, knn_pos, node_num)
# plot_graph_and_signal(er_adj, er_f, er_pos, node_num)
# plot_graph_and_signal(ba_adj, ba_f, ba_pos, node_num)

###
real_adj=rbf_adj
signal=rbf_signal
signal_num=signal.shape[0]
mask=np.random.uniform(size=(signal_num, node_num))
mask=mask<0.9
error_sigma=0.1
noise=signal_noise(signal_num, node_num, scale=error_sigma)

#noisy_signal=mask*signal
noisy_signal=signal+noise
noisy_signal=np.reshape(noisy_signal, (signal_num,node_num))

# ### KNN
# learned_knn_adj, learned_knn_lap=learn_knn_graph(noisy_signal, node_num)
# learned_knn_signal=learn_knn_signal(learned_knn_adj, noisy_signal, signal_num, node_num)


# iteration=5
# p_adj, p_signal=Primal_dual_gl_loop(node_num, noisy_signal, iteration, alpha=1, beta=1, theta=0.01, step_size=0.05)
# #s_adj, s_signal=Siprep_gl_loop(node_num, rgg_adj, noisy_signal, iteration, alpha=30, beta=0.2, theta=0.01, step_size=0.05)
# p_adj_filtered=filter_graph_to_knn(p_adj, node_num)
# #s_adj_filtered=filter_graph_to_knn(s_adj, node_num)
# real_adj_filtered=filter_graph_to_knn(real_adj, node_num)


# plot_graph_and_signal(real_adj_filtered, signal[0], rbf_pos, node_num)
# plot_graph_and_signal(real_adj_filtered, noisy_signal[0], rbf_pos, node_num)
# plot_graph_and_signal(p_adj_filtered, p_signal[0], rbf_pos, node_num)
# #plot_graph_and_signal(s_adj_filtered, s_signal[0], rgg_pos, node_num)
# plot_graph_and_signal(learned_knn_adj, learned_knn_signal[0], rbf_pos, node_num)

gl_adj_error_list=[]
gl_signal_error_list=[]
knn_adj_error_list=[]
knn_signal_error_list=[]
x_ticks=[]
for n in np.arange(1, signal_num, 5):
	print('Signal Number ~~~~~~~~~~~~~~~~~~ ', n)
	iteration=5
	input_signal=noisy_signal[:n]
	real_signal=signal[:n]
	signal_num=input_signal.shape[0]
	p_adj, p_signal=Primal_dual_gl_loop(node_num, input_signal, iteration, alpha=1, beta=0.1, theta=0.01, step_size=0.05)
	learned_knn_adj, learned_knn_lap=learn_knn_graph(input_signal, node_num)
	learned_knn_signal=learn_knn_signal(learned_knn_adj, input_signal, signal_num, node_num)

	gl_adj_error=np.linalg.norm(p_adj-real_adj)
	gl_signal_error=np.linalg.norm(p_signal-real_signal)
	knn_adj_error=np.linalg.norm(learned_knn_adj-real_adj)
	knn_signal_error=np.linalg.norm(learned_knn_signal-real_signal)

	gl_adj_error_list.extend([gl_adj_error])
	gl_signal_error_list.extend([gl_signal_error])
	knn_adj_error_list.extend([knn_adj_error])
	knn_signal_error_list.extend([knn_signal_error])
	x_ticks.extend([n])


newpath=path+'node_num_%s_signal_num_%s_error_%s'%(node_num, signal_num, int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

plt.plot(x_ticks, gl_adj_error_list, label='GL')
plt.plot(x_ticks, knn_adj_error_list, label='KNN')
plt.xlabel('Signal Number')
plt.ylabel('Total L2 Error')
plt.title('Graph Error')
plt.legend(loc=1)
plt.savefig(newpath+'adj_error_n_s_e_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100))+'.png', dpi=100)
plt.show()

plt.plot(x_ticks, gl_signal_error_list, label='GL')
plt.plot(x_ticks, knn_signal_error_list, label='KNN')
plt.xlabel('Signal Number')
plt.ylabel('Total L2 Error')
plt.title('Signal Error')
plt.legend(loc=1)
plt.savefig(newpath+'signal_error_n_s_e_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100))+'.png', dpi=100)
plt.show()


real_adj_filtered=filter_graph_to_knn(real_adj, node_num)
p_adj_filtered=filter_graph_to_knn(p_adj, node_num)
plot_graph_and_signal(real_adj_filtered, signal[0], rbf_pos, node_num,error_sigma, title='Real', path=newpath)
plot_graph_and_signal(real_adj_filtered, noisy_signal[0], rbf_pos, node_num,error_sigma, title='Noisy', path=newpath)
plot_graph_and_signal(p_adj_filtered, p_signal[0], rbf_pos, node_num,error_sigma, title='GL', path=newpath)
plot_graph_and_signal(learned_knn_adj, learned_knn_signal[0], rbf_pos, node_num,error_sigma, title='KNN', path=newpath)