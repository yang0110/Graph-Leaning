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
os.chdir('D:/Research/Graph Learning/code/')
from community import community_louvain
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns
from sklearn.datasets import make_blobs

def rbf_graph(node_num, dimension=2, threshold=0.5):
	features=np.random.uniform(low=0, high=1, size=(node_num, dimension))
	adj_matrix=rbf_kernel(features, gamma=(1)/(2*(0.5)**2))
	adj_matrix[np.where(adj_matrix<threshold)]=0.0
	np.fill_diagonal(adj_matrix,0)
	laplacian=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, laplacian, features


def knn_graph(node_num, dimension=2, k=10):
	features=np.random.uniform(low=0, high=1, size=(node_num, dimension))
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


def find_eigenvalues_matrix(eigen_values):
	eigenvalues_matrix=np.diag(np.sort(eigen_values))
	return eigenvalues_matrix

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix


def generate_signal_gl_siprep(signal_num, node_num, laplacian):
	mean=np.zeros(node_num)
	sigma_error=0.2
	pinv_lap=np.linalg.pinv(laplacian)
	cov=pinv_lap+sigma_error*np.identity(node_num)
	signals=np.random.multivariate_normal(mean, cov, size=signal_num)
	return signals

def generate_signal(signal_num, node_num, node_features, error_sigma):
	item_f=np.random.normal(size=(signal_num, node_features.shape[1]))
	signals=np.dot(node_features, item_f.T).T
	noise=np.random.normal(scale=error_sigma, size=(signals.shape[0], signals.shape[1]))
	noise_signals=signals+noise
	return signals, noise_signals, item_f

def make_blobs_signal(signal_num, node_num, cluster_num, dimension, error_sigma):
	node_features, labels=make_blobs(n_samples=node_num, n_features=dimension, centers=cluster_num, cluster_std=2, center_box=(0,1))
	item_f=np.random.normal(size=(signal_num, node_features.shape[1]))
	signals=np.dot(node_features, item_f.T).T
	noise=np.random.normal(scale=error_sigma, size=(signals.shape[0], signals.shape[1]))
	noise_signals=signals+noise
	adj_matrix=rbf_kernel(node_features)

	np.fill_diagonal(adj_matrix,0)
	return adj_matrix, node_features, signals, noise_signals

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

def f1(x,y):
	return np.sin((2-x-y)**2)

def f2(x,y):
	return np.cos((x+y)**2)

def f3(x,y):
	return  (x-0.5)**2+(y-0.5)**3+x-y 

def f4(x,y):
	return np.sin(3*(x-0.5)**2+(y-0.5)**2)
