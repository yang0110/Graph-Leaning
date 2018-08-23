import numpy as np
import random
from random import choice
import datetime
from matplotlib.pylab import *
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from community import community_louvain
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
#import seaborn as sns
from sklearn.datasets import make_blobs
from utils import *

def RGG(node_num, dimension=2):
	RS=np.random.RandomState(seed=100)
	features=RS.uniform(low=0, high=1, size=(node_num, dimension))
	adj_matrix=rbf_kernel(features, gamma=(1)/(2*(0.5)**2))
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian, features

def rbf_graph(node_num, dimension=2, threshold=0.5):
	RS=np.random.RandomState(seed=100)
	features=RS.uniform(low=0, high=1, size=(node_num, dimension))
	adj_matrix=rbf_kernel(features, gamma=(1)/(2*(0.5)**2))
	adj_matrix[adj_matrix<threshold]=0.0
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian, features

def knn_graph(node_num, dimension=2, k=5):
	RS=np.random.RandomState(seed=100)
	features=RS.uniform(low=0, high=1, size=(node_num, dimension))
	adj_matrix=rbf_kernel(features, gamma=(1)/(2*(0.5)**2))
	for i in range(node_num):
		rbf_row=adj_matrix[i,:]
		neighbors=np.argsort(rbf_row)[:node_num-k]
		adj_matrix[i, neighbors]=0
		adj_matrix[neighbors,i]=0
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian, features

def er_graph(node_num, prob=0.2, seed=2018):
	graph=nx.erdos_renyi_graph(node_num, prob, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian


def ba_graph(node_num, seed=2018):
	graph=nx.barabasi_albert_graph(node_num, m=1, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian	


def generate_signal_gl_siprep(signal_num, node_num, laplacian):
	mean=np.zeros(node_num)
	pinv_lap=np.linalg.pinv(laplacian)
	cov=pinv_lap
	signals=np.random.multivariate_normal(mean, cov, size=signal_num)
	return signals

def generate_signal(signal_num, node_num, node_pos):
	# linear combination of item feature and node feature
	RS=np.random.RandomState(seed=100)
	item_f=RS.normal(size=(signal_num, node_pos.shape[1]))
	signals=np.dot(node_pos, item_f.T).T
	return signals


def f1(x,y):
	return np.sin((2-x-y)**2)

def f2(x,y):
	return np.cos((x+y)**2)

def f3(x,y):
	return  (x-0.5)**2+(y-0.5)**3+x-y 

def f4(x,y):
	return np.sin(3*(x-0.5)**2+(y-0.5)**2)

def f5(x,y):
	return (x-0.5)+(y-0.5)

def Tikhonov_filter(x, alpha=10):
	return 1/(1+alpha*x)

def Heat_diffusion_filter(x, t=10):
	return np.exp(-t*x)

def Generative_model_filter(x):
	if x>0:
		y=1/np.sqrt(x)
	else:
		y=0
	return y

def original_signal(signal_num, node_num):
	signal=np.random.normal(loc=1.0, size=(signal_num, node_num))
	return signal

def Tikhonov_signal(original_signal, adj_matrix):
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	laplacian=laplacian/np.linalg.norm(laplacian)
	eigenvalues, eigenvectors=np.linalg.eig(laplacian)
	filtered_signal=[]
	for j in range(len(original_signal)):
		a=0
		for i in range(len(eigenvalues)):
			a+=eigenvectors[i]*Generative_model_filter(eigenvalues[i])*np.dot(eigenvectors[i], original_signal[j])
		filtered_signal.append(a)
	return np.array(filtered_signal)

def Heat_diffusion_signal(original_signal, adj_matrix):
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	laplacian=laplacian/np.linalg.norm(laplacian)
	eigenvalues, eigenvectors=np.linalg.eig(laplacian)
	resulted_signal=[]
	for j in range(len(original_signal)):
		a=0
		for i in range(len(eigenvalues)):
			a+=eigenvectors[i]*Heat_diffusion_filter(eigenvalues[i])*np.dot(eigenvectors[i], original_signal[j])
		resulted_signal.append(a)
	return np.array(resulted_signal)


def Generative_model_signal(original_signal, adj_matrix):
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	laplacian=laplacian/np.linalg.norm(laplacian)
	eigenvalues, eigenvectors=np.linalg.eig(laplacian)
	filtered_signal=[]
	for j in range(len(original_signal)):
		a=0
		for i in range(len(eigenvalues)):
			a+=eigenvectors[i]*Generative_model_filter(eigenvalues[i])*np.dot(eigenvectors[i], original_signal[j])
		filtered_signal.append(a)
	return np.array(filtered_signal)


def find_corrlation_matrix(signals):
	corr_matrix=np.corrcoef(signals.T)
	return corr_matrix

def create_networkx_graph(node_num, adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(node_num)))
	for i in range(node_num):
		for j in range(node_num):
			if adj_matrix[i,j]!=0:
				G.add_edge(i,j,weight=adj_matrix[i,j])
			else:
				pass
	return G

def find_eigenvalues_matrix(eigen_values):
	eigenvalues_matrix=np.diag(np.sort(eigen_values))
	return eigenvalues_matrix

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix

def learn_knn_graph(signals, node_num, k=5):
	print('Learning KNN Graph')
	adj=rbf_kernel(signals.T)
	np.fill_diagonal(adj,0)
	knn_adj=filter_graph_to_knn(adj, node_num, k=k)
	knn_lap=csgraph.laplacian(knn_adj, normed=False)
	return knn_adj, knn_lap

def learn_knn_signal(adj, signals, signal_num, node_num):
	print('Learning KNN Signals')
	new_signals=np.zeros((signal_num, node_num))
	for i in range(signal_num):
		for j in range(node_num):
			rbf_row=adj[j,:]
			neighbors=rbf_row>0
			weights=rbf_row[rbf_row>0]
			if len(weights)==0:
				new_signals[i,j]=signals[i,j]
			else:
				new_signals[i,j]=np.average(signals[i][neighbors], weights=weights)
	return new_signals

def signal_noise(signal_num, node_num, scale):
	RS=np.random.RandomState(seed=100)
	noise=RS.normal(scale=scale, size=(signal_num, node_num))
	return noise

# adj, f=RGG(10)
# laplacian=csgraph.laplacian(adj, normed=False)
# laplacian=laplacian/np.linalg.norm(laplacian)
# eigenvalues, eigenvectors=np.linalg.eig(laplacian)


# y=Tikhonov_signal(x, adj)
# y=Generative_model_signal(x, adj)
# y=Heat_diffusion_signal(x, adj)