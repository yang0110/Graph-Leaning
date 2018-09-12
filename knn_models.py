import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import networkx as nx 
import os 
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from utils import *
from synthetic_data import *



def KNN_graph(signals, node_num, k):
	W=rbf_kernel(signals.T)
	np.fill_diagonal(W,0)
	V=np.ones((node_num, node_num))
	for i in range(node_num):
		rbf_row=W[i,:]
		neighbors=np.argsort(rbf_row)[:node_num-k]
		V[i,neighbors]=0
	np.fill_diagonal(V,0)
	U=np.fmax(V, V.T)
	A=U*W
	return A

def Mutual_KNN_graph(signals, node_num, k):
	W=rbf_kernel(signals.T)
	np.fill_diagonal(W,0)
	V=np.ones((node_num, node_num))
	for i in range(node_num):
		rbf_row=W[i,:]
		neighbors=np.argsort(rbf_row)[:node_num-k]
		V[i,neighbors]=0
	np.fill_diagonal(V,0)
	U=np.fmin(V, V.T)
	A=U*W
	return A
	

def Centered_KNN_graph(signals, node_num, k):
	old_W=rbf_kernel(signals.T)
	ones=np.ones((node_num,1))
	a=np.identity(node_num)-(1/node_num)*(np.dot(ones, ones.T))
	W=a*old_W*a
	V=np.ones((node_num, node_num))
	for i in range(node_num):
		rbf_row=W[i,:]
		neighbors=np.argsort(rbf_row)[:node_num-k]
		V[i,neighbors]=0
	np.fill_diagonal(V,0)
	U=np.fmax(V, V.T)
	A=U*W
	return A

	