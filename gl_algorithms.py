import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
#import seaborn as sns
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from utils import *
from pygsp import graphs, plotting, filters
import pyunlocbox
import networkx as nx 
from gl_sigrep import Gl_sigrep


def Primal_dual_gl_loop(node_num, signal, iteration, alpha=1, beta=0.2, theta=0.01, step_size=0.5):
	for i in range(iteration):
		print('Primal Dual GL Time ~~~~~~~~~', i)
		Z=euclidean_distances(signal.T, squared=True)
		np.fill_diagonal(Z,0)
		Z=norm_W(Z, node_num)
		primal_gl=Primal_dual_gl(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
		primal_adj, error=primal_gl.run()
		print('graph error', error[-1])
		lap=csgraph.laplacian(primal_adj, normed=False)
		signal=np.dot(signal, np.linalg.inv((np.identity(node_num)+theta*lap)))
	return primal_adj, signal


def Siprep_gl_loop(node_num, signal, iteration, alpha=10, beta=0.2, theta=0.01, step_size=0.5):
	for i in range(iteration):
		print('GL SipRep Time ~~~~~~~~~~~', i)
		Z=euclidean_distances(signal.T, squared=True)
		np.fill_diagonal(Z,0)
		Z=norm_W(Z, node_num)
		gl=Gl_sigrep(node_num, Z, alpha=alpha, beta=beta, step_size=step_size)
		learned_adj, error=gl.run()
		print('graph error', error[-1])
		lap=csgraph.laplacian(learned_adj, normed=False)
		signal=np.dot(signal, np.linalg.inv((np.identity(node_num)+theta*lap)))
	return learned_adj, signal