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
	def __init__(self, node_num, Z, alpha, beta, iteration, c=0):
		self.node_num=node_num
		self.ncols=int(node_num*(node_num-1)/2)
		self.Z=Z
		self.z=vector_form(Z, node_num)
		self.alpha=alpha
		self.beta=beta
		self.W=np.zeros((node_num,node_num))
		self.w=np.zeros(int((node_num-1)*node_num/2))
		self.w_0=np.zeros(int((node_num-1)*node_num/2))
		self.c=c
		self.S=sum_squareform(node_num)
		self.d=np.dot(self.S, self.w)
		self.eplison=10**(-5)
		self.iteration=iteration
		self.y=None
		self.y_bar=None
		self.p=None
		self.p_bar=None
		self.q=None
		self.q_bar=None
		self.max_w=np.inf
		self.mu=2*(self.beta+self.c)+np.sqrt(2*(self.node_num-1))
		self.ep=lin_map(0.0,[0,1/(1+self.mu)], [0,1])
		self.gamma=lin_map(0.5, [self.ep, (1-self.ep)/self.mu], [0,1])
	def run(self):
		for i in range(self.iteration):
			print('iteration', i)

			self.y=self.w-self.gamma*(2*((self.beta+self.c)*self.w-self.c*self.w_0)+np.dot(self.S.T, self.d))
			self.y_bar=self.d+self.gamma*(self.d)

			print('self.y', self.y)
			print('self.y_bar', self.y_bar)
			temp1=np.fmax(0,self.y-2*self.gamma*self.z)
			print('temp1', temp1)
			temp2=np.fmin(self.max_w, temp1)
			print('temp2', temp2)
			self.p=temp2
			print('self.p', self.p)
			self.p_bar=(self.y_bar-np.sqrt((self.y_bar)**2+4*self.alpha*self.gamma))/2.0
			self.q=self.p-self.gamma*(2*((self.beta+self.c)*self.p-self.c*self.w_0)+np.dot(self.S.T, self.p_bar))
			self.q_bar=self.p_bar+self.gamma*(np.dot(self.S, self.p))
			print('self.p_bar', self.p_bar)
			print('self.q', self.q)
			print('self.q_bar', self.q_bar)


			w_i_1=self.w.copy()
			print('self.w', self.w)
			self.w=w_i_1-self.y+self.q
			print('self.w', self.w)

			print('self.d', self.d)
			d_i_1=self.d.copy()
			self.d=d_i_1-self.y_bar+self.q_bar
			print('self.d', self.d)

			w_diff=np.linalg.norm(self.w-w_i_1)
			w_ratio=w_diff/np.linalg.norm(w_i_1)
			d_diff=np.linalg.norm(self.d-d_i_1)
			d_ratio=d_diff/np.linalg.norm(d_i_1)
			if (w_ratio<self.eplison) and (d_ratio<self.eplison):
				break 
			else:
				pass 
			index1=np.tril_indices(self.node_num, -1)
			index2=np.triu_indices(self.node_num,1)
			self.W[index1]=self.w
			self.W[index2]=self.w

			print('w_ratio', w_ratio)
			print('w_diff', w_diff)
			print('norm(w_i_1)', np.linalg.norm(w_i_1))
			print('d_ratio', d_ratio)
			print('d_diff', d_diff)
			print('norm(d_i_1)', np.linalg.norm(d_i_1))

		return self.w, self.W
