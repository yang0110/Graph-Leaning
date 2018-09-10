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
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from collections import Counter
from scipy.sparse import csgraph
#import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain
from pygsp import graphs, plotting, filters
import pyunlocbox


def sum_squareform(n):
	#sum operator that find degree from upper triangle
	ncols=int((n-1)*n/2)
	I=np.zeros(ncols)
	J=np.zeros(ncols)
	k=0
	for i in list((np.arange(n)+1))[1:]:

		I[k:k+(n-i)+1]=(np.arange(n)+1)[i-1:]
		k=k+(n-i+1)

	k=0
	for i in list((np.arange(n)+1))[1:]:
		J[k:k+(n-i)+1]=i-1
		k=k+n-i+1

	I=I.astype(int)
	J=J.astype(int)
	ys=list(I)+list(J)
	xs=list(np.arange(ncols)+1)+list(np.arange(ncols)+1)
	s_T=np.zeros((ncols,n))
	for i in range(len(ys)):
		s_T[(xs[i]-1),(ys[i]-1)]=1

	return s_T.T

def vector_form(W,n):
	w=W[np.triu_indices(n,1)]
	# the triangle-upper 
	return w

def matrix_form(w, n):
	W=np.zeros((n,n))
	W[np.triu_indices(n,1)]=w
	for i in range(n):
		for j in range(n):
			W[j,i]=W[i,j]
	return W

def scale_0_1(w):
	mms=MinMaxScaler()
	norm_w=mms.fit_transform(w.reshape(-1,1))
	return norm_w.ravel()

def norm_W(W,n):
	w=vector_form(W,n)
	norm_w=scale_0_1(w)
	norm_W=matrix_form(norm_w,n)
	return norm_W

def lin_map(x, lims_out, lims_in):
	a=lims_in[0]
	b=lims_in[1]
	c=lims_out[0]
	d=lims_out[1]
	y=np.zeros(len([x]))
	y=((x-a)*(d-c)/(b-a))+c
	return y


def filter_graph_to_knn(adj_matrix, node_num, k):
	a=adj_matrix.copy()
	for i in range(node_num):
		rbf_row=a[i,:].ravel()
		neighbors=np.argsort(rbf_row)[:node_num-k]
		a[i, neighbors]=0
		a[neighbors,i]=0
	np.fill_diagonal(a,0)
	return a


def filter_graph_to_rbf(adj_matrix, node_num, thres=0.5):
	adj_matrix[adj_matrix<thres]=0
	np.fill_diagonal(adj_matrix,0)
	return adj_matrix


def calculate_smoothness(signal, laplacian):
	if len(signal)==1:
		smooth=np.dot(signal, np.dot(laplacian, signal.T))
	else:
		smooth=[]
		for i in range(len(signal)):
			a=signal[i]
			s=np.dot(a, np.dot(laplacian, a.T))
			smooth.extend([s])
	return smooth


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

def plot_graph_and_signal(adj_matrix, signal, pos, node_num, error_sigma, title='Graph', path='newpath', show=True):
	graph=create_networkx_graph(node_num, adj_matrix)
	edge_weight=adj_matrix[np.triu_indices(node_num, 1)]
	edge_color=edge_weight[edge_weight>0]
	if len(np.unique(edge_color))==1: # unweighted
		nodes=nx.draw_networkx_nodes(graph, pos, node_color=signal, node_size=100, cmap=plt.cm.jet)
		edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.3, edge_color='b')
	else:# weighted
		nodes=nx.draw_networkx_nodes(graph, pos, node_color=signal, node_size=100, cmap=plt.cm.jet)
		edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues, vmin=0, vmax=1)
	plt.axis('off')
	plt.title(title)
	plt.savefig(path+title+'.png', dpi=200)
	if show==True:
		plt.show()
	else:
		plt.clf()

def generate_graph_from_rbf(adj_matrix):
	adj_matrix=np.matrix(adj_matrix)
	G=nx.from_numpy_matrix(adj_matrix)
	return G

def generate_graph(adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(adj_matrix.shape[0])))
	for i in range(adj_matrix.shape[0]):
		for j in range(adj_matrix.shape[1]):
			if adj_matrix[i,j]==0.0:
				pass 
			else:
				G.add_edge(i,j, weight=adj_matrix[i,j])
	return G


def find_community_best_partition(graph):
	parts=community_louvain.best_partition(graph)
	values=[parts.get(node) for node in graph.nodes()]
	clusters=values
	n_clusters=len(np.unique(values))
	return clusters, n_clusters

def total_variation_signal_learning(adj, noisy_signal, gamma=3.0):
	G=graphs.Graph(adj)
	gamma=gamma
	d=pyunlocbox.functions.dummy()
	r=pyunlocbox.functions.norm_l1()
	f=pyunlocbox.functions.norm_l2(w=1, y=noisy_signal, lambda_=gamma)
	G.compute_differential_operator()
	L=G.D.toarray()
	step=0.999/(1+np.linalg.norm(L))
	solver=pyunlocbox.solvers.mlfbf(L=L, step=step)
	x0=noisy_signal.copy()
	prob=pyunlocbox.solvers.solve([d,r,f], solver=solver, x0=x0, rtol=0, maxit=1000)
	sol=prob['sol']
	return sol 
