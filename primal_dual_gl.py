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
from utils import sum_squareform, vector_form, lin_map



class Primal_dual_gl():
	def __init__(self, node_num, Z, alpha=0.1, beta=1, step_size=0.5):
		self.node_num=node_num
		self.ncols=int(node_num*(node_num-1)/2)
		self.Z=Z
		self.z=vector_form(Z, node_num)
		self.alpha=alpha
		self.beta=beta
		self.W=np.zeros((node_num,node_num))
		self.w=np.zeros(self.ncols)
		self.w_0=np.zeros(self.ncols)
		self.S=sum_squareform(node_num)
		self.d=np.dot(self.S, self.w)
		self.step_size=step_size
		self.eplison=10**(-5)
		self.max_iteration=3000
		self.y=None
		self.y_bar=None
		self.p=None
		self.p_bar=None
		self.q=None
		self.q_bar=None
		self.c=0
		self.max_w=1.0
		self.mu=2*(self.beta+self.c)+np.sqrt(2*(self.node_num-1))
		self.ep=lin_map(0.0, [0,1/(1+self.mu)], [0,1])
		self.gamma=lin_map(self.step_size, [self.ep, (1-self.ep)/self.mu], [0,1])

	def run(self, real_w):
		error_list=[]
		for i in range(self.max_iteration):
			#print('iteration', i)

			self.y=self.w-self.gamma*(2*((self.beta+self.c)*self.w-self.c*self.w_0)+np.dot(self.S.T, self.d))
			self.y_bar=self.d+self.gamma*(self.d)

			self.p=np.fmin(self.max_w, np.fmax(0, self.y-2*self.gamma*self.z))
			self.p_bar=(self.y_bar-np.sqrt((self.y_bar)**2+4*self.alpha*self.gamma))/2.0
			self.q=self.p-self.gamma*(2*((self.beta+self.c)*self.p-self.c*self.w_0)+np.dot(self.S.T, self.p_bar))
			self.q_bar=self.p_bar+self.gamma*(np.dot(self.S, self.p))

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

			error=np.linalg.norm(self.W-real_w)
			error_list.extend([error])
		return self.W, error_list
