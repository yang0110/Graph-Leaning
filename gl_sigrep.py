### GL-siprep is the graph learning model proposed by xiaowen dong 
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
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns
from synthetic_data import *
from scipy.optimize import minimize
from utils import sum_squareform, vector_form, lin_map,filter_graph_to_knn



class Gl_sigrep():
	def __init__(self, node_num, Z, alpha=0.1, beta=0.1, c=0):
		self.node_num=node_num
		self.ncols=int(node_num*(node_num-1)/2)
		self.Z=Z
		self.z=vector_form(Z, node_num)
		self.alpha=alpha
		self.beta=beta
		self.W=np.zeros((node_num,node_num))
		self.w=np.zeros(self.ncols)
		self.w_0=np.zeros(self.ncols)
		self.c=c
		self.S=sum_squareform(node_num)
		self.K=2*np.ones(self.ncols)
		self.d=np.dot(self.K, self.w)
		self.eplison=10**(-5)
		self.max_iteration=2000
		self.y=None
		self.y_bar=None
		self.p=None
		self.p_bar=None
		self.q=None
		self.q_bar=None
		self.max_w=np.inf
		self.mu=2*self.alpha*self.node_num+2*np.sqrt(self.ncols)
		self.ep=lin_map(0.0, [0,1/(1+self.mu)], [0,1])
		self.gamma=lin_map(0.5, [self.ep, (1-self.ep)/self.mu], [0,1])

	def run(self, real_w):
		error_list=[]
		for i in range(self.max_iteration):
			#print('iteration', i)

			self.y=self.w-self.gamma*(2*self.alpha*(2*self.w+np.dot(self.S.T, np.dot(self.S, self.w)))+2*self.d)
			self.y_bar=self.d+self.gamma*(2*np.sum(self.w))

			self.p=np.fmin(self.max_w, np.fmax(0, self.y-2*self.gamma*self.z))
			self.p_bar=self.y_bar-self.gamma*self.node_num
			self.q=self.p-self.gamma*(self.alpha*(2*self.p+np.dot(self.S.T, np.dot(self.S, self.p)))+2*self.p)
			self.q_bar=self.p_bar+self.gamma*(2*np.sum(self.p))

			w_i_1=self.w.copy()
			self.w=self.w-self.y+self.q
#			print('self.w', self.w)

			d_i_1=self.d.copy()
			self.d=self.d-self.y_bar+self.q_bar

			w_diff=np.linalg.norm(self.w-w_i_1)
			w_ratio=w_diff/np.linalg.norm(w_i_1)
			d_diff=np.linalg.norm(self.d-d_i_1)
			d_ratio=d_diff/np.linalg.norm(d_i_1)
			if (w_ratio<self.eplison) and (d_ratio<self.eplison):
			 	break 
			else:
				pass 
			index=np.triu_indices(self.node_num,1)
			self.W[index]=self.w
			for i in range(self.node_num):
				for j in range(self.node_num):
					self.W[j,i]=self.W[i,j]

#			print('w_ratio', w_ratio)
#			print('d_ratio', d_ratio)
			error=np.linalg.norm(self.W-real_w)
			error_list.extend([error])
		return self.W, error_list
