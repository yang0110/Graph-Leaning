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

class Gl_sigrep():
	def __init__(self, node_num, input_, iteration, alpha_, beta_):
		self.node_num=node_num
		self.input=input_
		self.iteration=iteration
		self.alpha_=alpha_
		self.beta_=beta_
		self.output=input_
		self.laplacian=np.random.normal(size=(self.node_num, self.node_num))

	def update_laplacian_objective(self, x):
		x=x.reshape((self.node_num, self.node_num))
		a=np.dot(self.output, x)
		b=np.dot(a, self.output.T)
		c=np.trace(b)
		d=self.alpha_*c 
		e=np.linalg.norm(x, 'fro')**2
		f=self.beta_*e
		h=d+f
		return h

	def cons_1(self, x):
		x=x.reshape((self.node_num, self.node_num))

		return np.trace(x)-self.node_num

	def cons_2(self, x):
		x=x.reshape((self.node_num, self.node_num))
		return np.dot(x, np.ones((self.node_num)))-np.zeros((self.node_num))

	def cons_3(self, x):
		x=x.reshape((self.node_num, self.node_num))
		x_t=x.T
		return np.sum(x-x_t)

	def update_output_objective(self, x):
		x=x.reshape((len(self.input),self.node_num))
		a=np.dot(x, self.laplacian)
		b=np.dot(a, x.T)
		c=np.trace(b)
		d=self.alpha_*c
		e=self.input-x
		f=np.linalg.norm(e, 'fro')
		g=f**2
		h=g+d
		return h 




	def run(self):
		for i in range(self.iteration):
			print('iteration ~~~~~~~~~ ', i)
			cons=({'type': 'eq', 'fun': self.cons_1},
				{'type': 'eq', 'fun': self.cons_2},
				{'type': 'eq', 'fun': self.cons_3}
				)

			res1=minimize(self.update_laplacian_objective, self.laplacian, constraints=cons, options={'gtol':1e-4, 'disp': True} )

			self.laplacian=res1.x.reshape((self.node_num, self.node_num))

			res2=minimize(fun=self.update_output_objective, x0=self.output)

			self.output=res2.x.reshape((len(self.input), self.node_num))

		return self.laplacian, self.output