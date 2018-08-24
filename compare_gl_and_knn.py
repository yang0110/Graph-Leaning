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
error_sigma=0.1

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

###
rgg_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rgg_lap, error_sigma)
rbf_gl_signal=generate_signal_gl_siprep(signal_num, node_num, rbf_lap, error_sigma)
knn_gl_signal=generate_signal_gl_siprep(signal_num, node_num, knn_lap, error_sigma)
er_gl_signal=generate_signal_gl_siprep(signal_num, node_num, er_lap, error_sigma)
ba_gl_signal=generate_signal_gl_siprep(signal_num, node_num, ba_lap, error_sigma)
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

newpath=path+'node_num_%s_signal_num_%s_error_%s'%(node_num, signal_num, int(error_sigma*100))+str(timeRun)+'/'
if not os.path.exists(newpath):
	    os.makedirs(newpath)

gl_adj_error_list=[]
gl_signal_error_list=[]
x_ticks=[]
for n in np.arange(1, signal_num, 5):
	print('Signal Number ~~~~~~~~~~~~~~~~~~ ', n)
	iteration=5
	input_signal=noisy_signal[:n]
	real_signal=signal[:n]
	p_adj, p_signal=Primal_dual_gl_loop(node_num, input_signal, iteration, alpha=1, beta=0.1, theta=0.01, step_size=0.05)
	gl_adj_error=np.linalg.norm(p_adj-real_adj)
	gl_signal_error=np.linalg.norm(p_signal-real_signal)

	gl_adj_error_list.extend([gl_adj_error])
	gl_signal_error_list.extend([gl_signal_error])

	x_ticks.extend([n])




knn_adj_error_df=pd.DataFrame(index=np.arange(1, signal_num, 5), columns=list(np.arange(5,node_num,5)))
knn_signal_error_df=pd.DataFrame(index=np.arange(1, signal_num, 5), columns=list(np.arange(5,node_num,5)))
knn_adj_s={}
knn_signal_s={}
for k in np.arange(5,node_num,5):
	k=k
	knn_adj_error_list=[]
	knn_signal_error_list=[]
	for n in np.arange(1, signal_num, 5):
		input_signal=noisy_signal[:n]
		real_signal=signal[:n]
		signal_length=input_signal.shape[0]
		learned_knn_adj, learned_knn_lap=learn_knn_graph(input_signal, node_num, k=k)
		learned_knn_signal=learn_knn_signal(learned_knn_adj, input_signal, signal_length, node_num)
		knn_adj_error=np.linalg.norm(learned_knn_adj-real_adj)
		knn_signal_error=np.linalg.norm(learned_knn_signal-real_signal)
		knn_adj_error_list.extend([knn_adj_error])
		knn_signal_error_list.extend([knn_signal_error])
	knn_adj_error_df[k]=knn_adj_error_list
	knn_signal_error_df[k]=knn_signal_error_list
	knn_adj_s[k]=learned_knn_adj
	knn_signal_s[k]=learned_knn_signal
	if k==5 or k==10:
		plot_graph_and_signal(learned_knn_adj, learned_knn_signal[0], rbf_pos, node_num, error_sigma, title='KNN', path=newpath+'n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k), show=False)
		normed_knn_adj=norm_W(learned_knn_adj, node_num)
		plt.pcolor(normed_knn_adj, cmap='RdBu')
		plt.title('KNN Adj')
		plt.colorbar()
		plt.savefig(newpath+'knn_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
		plt.clf()
	else:
		pass



knn_adj_error_df.plot()
plt.title('KNN Graph Error')
plt.legend(loc=1)
plt.ylabel('Total L2 Error')
plt.savefig(newpath+'knn_graph_error_n_s_e_%s_%s_%s'%(node_num, signal_num, error_sigma)+'.png', dpi=100)
plt.clf()

knn_signal_error_df.plot()
plt.title('KNN Signal Error')
plt.legend(loc=1)
plt.ylabel('Total L2 Error')
plt.savefig(newpath+'knn_signal_error_n_s_e_%s_%s_%s'%(node_num, signal_num, error_sigma)+'.png', dpi=100)
plt.clf()

np.save(newpath+'signal'+'.npy',signal)
np.save(newpath+'noisy_signal'+'.npy',noisy_signal)
np.save(newpath+'ground_truth_adj'+'.npy', rbf_adj)
np.save(newpath+'filtered_ground_truth_adj'+'.npy',  filter_graph_to_knn(rbf_adj, node_num))

knn_adj_error_df.to_csv(newpath+'knn_adj_error', index=False)
knn_signal_error_df.to_csv(newpath+'knn_signal_error', index=False)
np.save(newpath+'knn_learned_adj'+'.npy',knn_adj_s)
np.save(newpath+'knn_learned_signal'+'.npy',knn_signal_s)

np.save(newpath+'gl_learned_adj'+'.npy', p_adj)
np.save(newpath+'gl_learned_signal'+'.npy', p_signal)
np.save(newpath+'gl_adj_error_list'+'.npy', gl_adj_error_list)
np.save(newpath+'gl_signal_error_list'+'.npy', gl_signal_error_list)



plt.plot(x_ticks, gl_adj_error_list, label='GL')
plt.plot(x_ticks, knn_adj_error_list, label='KNN')
plt.xlabel('Signal Number')
plt.ylabel('Total L2 Error')
plt.title('Graph Error')
plt.legend(loc=1)
plt.savefig(newpath+'graph_error_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100),k)+'.png', dpi=100)
plt.clf()

plt.plot(x_ticks, gl_signal_error_list, label='GL')
plt.plot(x_ticks, knn_signal_error_list, label='KNN')
plt.xlabel('Signal Number')
plt.ylabel('Total L2 Error')
plt.title('Signal Error')
plt.legend(loc=1)
plt.savefig(newpath+'signal_error_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100),k)+'.png', dpi=100)
plt.clf()


real_adj_filtered=filter_graph_to_knn(real_adj, node_num)
p_adj_filtered=filter_graph_to_knn(p_adj, node_num)
plot_graph_and_signal(real_adj_filtered, signal[0], rbf_pos, node_num,error_sigma, title='Real', path=newpath+'n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k), show=False)
plot_graph_and_signal(real_adj_filtered, noisy_signal[0], rbf_pos, node_num,error_sigma, title='Noisy', path=newpath+'n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k), show=False)
plot_graph_and_signal(p_adj_filtered, p_signal[0], rbf_pos, node_num,error_sigma, title='GL', path=newpath+'n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k), show=False)
plot_graph_and_signal(learned_knn_adj, learned_knn_signal[0], rbf_pos, node_num,error_sigma, title='KNN', path=newpath+'n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k), show=False)


normed_real_adj=norm_W(real_adj, node_num)
normed_filtered_real_adj=norm_W(real_adj_filtered, node_num)
normed_gl_adj=norm_W(p_adj, node_num)
normed_filtered_gl_adj=norm_W(p_adj_filtered, node_num)
normed_knn_adj=norm_W(learned_knn_adj, node_num)

plt.figure(figsize=(5,5))
plt.pcolor(normed_real_adj, cmap='RdBu')
plt.title('True Adj')
plt.colorbar()
plt.savefig(newpath+'true_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
plt.clf()

plt.figure(figsize=(5,5))
plt.pcolor(normed_filtered_real_adj, cmap='RdBu')
plt.title('Filtered True Adj')
plt.colorbar()
plt.savefig(newpath+'filtered_true_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
plt.clf()

plt.figure(figsize=(5,5))
plt.pcolor(normed_gl_adj, cmap='RdBu')
plt.title('GL Adj')
plt.colorbar()
plt.savefig(newpath+'gl_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
plt.clf()

plt.figure(figsize=(5,5))
plt.pcolor(normed_filtered_gl_adj, cmap='RdBu')
plt.title('GL Adj')
plt.colorbar()
plt.savefig(newpath+'filtered_gl_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
plt.clf()

plt.figure(figsize=(5,5))
plt.pcolor(normed_knn_adj, cmap='RdBu')
plt.title('KNN Adj')
plt.colorbar()
plt.savefig(newpath+'knn_adj_n_s_e_k_%s_%s_%s_%s'%(node_num, signal_num, int(error_sigma*100), k)+'.png', dpi=200)
plt.clf()

