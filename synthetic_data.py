import numpy as np
import random
from random import choice
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from scipy.sparse import csgraph
from sklearn.datasets import make_blobs
from utils import *
from knn_models import *
from sklearn.preprocessing import normalize


def RGG(node_num, dimension=2):
	RS=np.random.RandomState(seed=100)
	features=RS.uniform(low=0, high=1, size=(node_num, dimension))
	adj_matrix=rbf_kernel(features, gamma=(1)/(2*(0.5)**2))
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian, features

def generate_rbf_graph(node_num, pos, threshold=0.75):
	RS=np.random.RandomState(seed=100)
	adj_matrix=rbf_kernel(pos, gamma=(1)/(2*(0.5)**2))
	adj_matrix[adj_matrix<threshold]=0.0
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian

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

def generate_er_graph(node_num, prob=0.2, seed=2018):
	graph=nx.erdos_renyi_graph(node_num, prob, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian


def generate_ba_graph(node_num, seed=2018):
	graph=nx.barabasi_albert_graph(node_num, m=1, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	laplacian=nx.laplacian_matrix(graph).toarray()
	return adj_matrix, laplacian


def generate_signal_gl_siprep(signal_num, node_num, laplacian, error_sigma):
	mean=np.zeros(node_num)
	normed_lap=normalized_trace(laplacian, node_num)
	pinv_lap=np.linalg.pinv(normed_lap)
	cov=pinv_lap
	signal=np.random.multivariate_normal(mean, cov, size=signal_num)
	noisy_cov=pinv_lap+error_sigma*np.identity(node_num)
	noisy_signal=np.random.multivariate_normal(mean, noisy_cov, size=signal_num)
	return noisy_signal, signal

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
				G.add_edge(i,j, weight=adj_matrix[i,j])
			else:
				pass
	return G

def find_eigenvalues_matrix(eigen_values):
	eigenvalues_matrix=np.diag(np.sort(eigen_values))
	return eigenvalues_matrix

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix

def learn_knn_graph_from_node_features(node_features, node_num, k=5):
	adj=rbf_kernel(node_features)
	np.fill_diagonal(adj,0)
	knn_adj=filter_graph_to_knn(adj, node_num, k=k)
	knn_lap=csgraph.laplacian(knn_adj, normed=False)
	return knn_adj, knn_lap

def learn_knn_graph(signals, node_num, k=5):
	#print('Learning KNN Graph')
	adj=rbf_kernel(signals.T)
	np.fill_diagonal(adj,0)
	knn_adj=filter_graph_to_knn(adj, node_num, k=k)
	knn_lap=csgraph.laplacian(knn_adj, normed=False)
	return knn_adj, knn_lap

def learn_knn_signal(adj, signals, signal_num, node_num):
	#print('Learning KNN Signals')
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


def blob_data(node_num, signal_num, dimension, cluster_num, cluster_std, noise_scale):
	x, y=make_blobs(n_samples=node_num, n_features=dimension, centers=cluster_num, cluster_std=cluster_std, center_box=(0,1.0), shuffle=False)
	x=MinMaxScaler().fit_transform(x)
	#item_f, item_y=make_blobs(n_samples=signal_num, n_features=dimension, centers=cluster_num, cluster_std=cluster_std, center_box=(0, 1.0), shuffle=False)
	item_f=np.random.uniform(size=(signal_num, dimension))
	#item_f=MinMaxScaler().fit_transform(item_f)
	#item_f=generate_item_features(signal_num, dimension)
	signal=np.dot(item_f, x.T)
	noise=np.random.normal(size=(signal_num, node_num), scale=noise_scale)
	noisy_signal=signal+noise
	return noisy_signal, signal, item_f, x, y

def generate_item_features(item_num, dimension):
	item_features=np.empty([item_num, dimension])
	for i in range(dimension):
		item_features[:,i]=np.random.normal(0, np.sqrt(1.0*(dimension-1)/dimension), item_num)
	item_features=MinMaxScaler().fit_transform(item_features)
	return item_features

def generate_all_random_users(iterations, user_num):
	random_users=np.random.choice(np.arange(user_num), size=iterations, replace=True)
	return random_users


def generate_all_article_pool(iterations, pool_size, article_num):
	all_article_pool=[]
	for i in range(iterations):
		pool=np.random.choice(np.arange(article_num), size=pool_size, replace=True)
		all_article_pool.append(pool)
	all_article_pool=np.array(all_article_pool)

	return all_article_pool


	
# adj, f=RGG(10)
# laplacian=csgraph.laplacian(adj, normed=False)
# laplacian=laplacian/np.linalg.norm(laplacian)
# eigenvalues, eigenvectors=np.linalg.eig(laplacian)


# y=Tikhonov_signal(x, adj)
# y=Generative_model_signal(x, adj)
# y=Heat_diffusion_signal(x, adj)