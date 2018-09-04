import numpy as np 
import pandas as pandas
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from utils import *
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

class GL_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size=0.5, true_user_features=None, true_graph=None):
		self.user_num=user_num
		self.item_num=item_num
		self.dimension=dimension
		self.item_pool_size=item_pool_size
		self.alpha=alpha
		self.gl_alpha=gl_alpha
		self.gl_beta=gl_beta
		self.gl_theta=gl_theta
		self.gl_step_size=gl_step_size
		self.noisy_signal_per_user={}
		self.noisy_signal=None
		self.denoised_signal=None
		self.denoised_signal_per_user={}
		self.picked_items_per_user={}
		self.learned_user_features=np.zeros((self.user_num, self.dimension))
		self.cov_matrix={}
		self.bias={}
		self.true_user_features=true_user_features
		self.item_features=None
		self.learned_graph=None
		self.true_graph=true_graph
		self.iteration=None
		self.noisy_signal=None
		self.avaiable_noisy_signal=None
		self.picked_items=[]
		self.adj_matrix=None
		self.learning_error=[]
		self.served_user_num=None
		self.served_user=[]
		self.cum_regret=[0]




	def pick_item_and_payoff(self, user, item_pool, time):

		mean=np.dot(self.item_features[item_pool], self.learned_user_features[user])
		temp1=np.dot(self.item_features[item_pool], np.linalg.inv(self.cov_matrix[user]))
		temp2=np.sum(temp1*self.item_features[item_pool], axis=1)*np.log(time+1)
		var=np.sqrt(temp2)
		pta=mean+self.alpha*var
		picked_item=item_pool[np.argmax(pta)]
		payoff=self.noisy_signal[picked_item, user]
		self.noisy_signal_per_user[user].extend([payoff])
		self.picked_items_per_user[user].extend([picked_item])
		self.picked_items.extend([picked_item])
		self.avaiable_noisy_signal=self.noisy_signal[self.picked_items]
		return picked_item, payoff


	def graph_and_signal_learning(self):

		Z=euclidean_distances(self.avaiable_noisy_signal.T, squared=True)
		np.fill_diagonal(Z,0)
		Z=norm_W(Z, self.user_num)
		primal_gl=Primal_dual_gl(self.user_num, Z, alpha=self.gl_alpha, beta=self.gl_beta, step_size=self.gl_step_size)
		primal_adj, error=primal_gl.run()
		lap=csgraph.laplacian(primal_adj, normed=False)
		self.denoised_signal=np.dot(self.avaiable_noisy_signal, np.linalg.inv((np.identity(self.user_num)+self.gl_theta*lap)))
		self.adj_matrix=primal_adj.copy()
		return self.adj_matrix, self.denoised_signal


	def update_user_feature(self, user, picked_item, payoff):
		#self.cov_matrix[user]+=np.outer(self.item_features[picked_item], self.item_features[picked_item])
		#self.bias[user]=self.item_features[picked_item]*self.denoised_signal[picked_item, user]
		#self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])
		item_f=self.item_features[self.picked_items_per_user[user]]
		#print('item_f', item_f)
		payoffs=self.noisy_signal_per_user[user]
		#print('payoffs', payoffs)
		t1=np.dot(item_f.T, item_f)
		#print('t1', t1)
		t2=np.dot(np.linalg.inv(t1), item_f.T)
		#print('t2', t2)
		t3=np.dot(t2, payoffs)
		self.learned_user_features[user]=t3


	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])



	def run(self, user_pool, item_pools, item_features, noisy_signal, iteration):
		self.noisy_signal=noisy_signal
		self.iteration=iteration
		self.item_features=item_features
		for i in range(self.iteration):
			print('GL MAB Iteration ~~~~~~~~~~~~', i)
			user=user_pool[i]
			item_pool=item_pools[i]
			if user in self.served_user:
				pass
			else:
				self.cov_matrix[user]=np.identity(self.dimension)
				self.bias[user]=np.zeros(self.dimension)
				self.picked_items_per_user[user]=[]
				self.noisy_signal_per_user[user]=[]
				self.denoised_signal_per_user[user]=[]
			self.served_user.extend([user])

			picked_item, payoff=self.pick_item_and_payoff(user, item_pool, i)
			self.graph_and_signal_learning()
			self.update_user_feature(user, picked_item, payoff)
			self.find_regret(user, item_pool, payoff)
			if self.true_user_features is not None:
				error=np.linalg.norm(self.learned_user_features-self.true_user_features)
				self.learning_error.extend([error])
			else:
				pass 

		return self.cum_regret,  self.adj_matrix, self.learned_user_features, self.learning_error, self.denoised_signal



