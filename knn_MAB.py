import numpy as np 
import pandas as pandas
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from utils import *
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


class KNN_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, K=10, jump_step=10, true_user_features=None, true_graph=None):
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=None
		self.dimension=dimension
		self.alpha=1+np.sqrt(np.log(2.0/alpha)/2.0)
		self.item_pool_size=item_pool_size
		self.K=K
		self.true_user_features=true_user_features
		self.true_graph=true_graph
		self.served_user=[]
		self.noisy_signal=None
		self.denoised_signal=None
		self.picked_items_per_user={}
		self.payoffs_per_user={}
		self.picked_items=[]
		self.iteration=None
		self.cov_matrix={}
		self.bias={}
		self.cluster_cov_matrix={}
		self.cluster_bias={}		
		self.learned_user_features=np.zeros((self.user_num, self.dimension))
		self.adj=np.identity(self.user_num)
		self.lap=None
		self.cum_regret=[0]
		self.learning_error=[]
		self.graph_error=[]
		self.true_signal=None
		self.jump_step=jump_step
		self.learned_cluster_features=np.zeros((self.user_num, self.dimension))
	def initial(self):
		for u in range(self.user_num):
			self.cov_matrix[u]=np.identity(self.dimension)
			self.bias[u]=np.zeros(self.dimension)
			self.cluster_cov_matrix[u]=np.identity(self.dimension)
			self.cluster_bias[u]=np.zeros(self.dimension)	

	def pick_item_and_payoff(self, user, item_pool, time):
		mean=np.dot(self.item_features[item_pool], self.learned_cluster_features[user])
		temp1=np.dot(self.item_features[item_pool], np.linalg.inv(self.cluster_cov_matrix[user]))
		temp2=np.sum(temp1*self.item_features[item_pool], axis=1)*np.log(time+1)
		var=np.sqrt(temp2)
		pta=mean+self.alpha*var
		picked_item=item_pool[np.argmax(pta)]
		payoff=self.noisy_signal[picked_item, user]
		self.payoffs_per_user[user].extend([payoff])
		self.picked_items_per_user[user].extend([picked_item])
		if picked_item not in self.picked_items:
			self.picked_items.extend([picked_item])
		else:
			pass
		self.avaiable_noisy_signal=self.noisy_signal[self.picked_items]
		return picked_item, payoff

	def knn_graph(self, time):
		if (time%self.jump_step==0):
			print('Update Graph')
			self.adj, self.lap=learn_knn_graph(self.avaiable_noisy_signal, self.user_num, k=self.K)
		else:
			pass

	def knn_graph_from_node_features(self, time):
		if (time%self.jump_step==0):
			print('Update Graph')
			self.adj, self.lap=learn_knn_graph_from_node_features(self.learned_user_features, self.user_num, k=self.K)
		else:
			pass

	def knn_signal(self):
		self.denoised_signal=learn_knn_signal(self.adj, self.avaiable_noisy_signal, len(self.avaiable_noisy_signal), self.user_num)

	def update_cluster_features(self, user):
		adj_copy=self.adj.copy()
		np.fill_diagonal(adj_copy,1)
		sum_cov_matrix=np.identity(self.dimension)
		sum_bais=np.zeros(self.dimension)
		for i_user in range(self.user_num):
			sum_cov_matrix+=(self.cov_matrix[i_user]-np.identity(self.dimension))
			sum_bais+=self.bias[i_user]
		self.cluster_cov_matrix[user]=sum_cov_matrix
		self.cluster_bias[user]=sum_bais
		weights=adj_copy[user]
		sum_weights=np.sum(weights)
		weights=weights/sum_weights
		self.learned_cluster_features[user]=np.average(self.learned_user_features, axis=0, weights=weights)

	def update_user_features(self, user, picked_item, payoff):
		item_f=self.item_features[picked_item]
		self.cov_matrix[user]+=np.outer(item_f, item_f)
		self.bias[user]+=item_f*payoff
		self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])

	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])

	def run(self, user_pool, item_pools, item_features, noisy_signal, true_signal, iteration):
		self.iteration=iteration
		self.noisy_signal=noisy_signal
		self.true_signal=true_signal
		self.item_features=item_features
		self.initial()
		for i in range(self.iteration):
			print('KNN MAB Time ~~~~~~~~~~~~ ', i)
			user=user_pool[i]
			item_pool=item_pools[i]
			if user not in self.served_user:
				self.picked_items_per_user[user]=[]
				self.payoffs_per_user[user]=[]
				self.served_user.extend([user])

			else:
				pass

			picked_item, payoff=self.pick_item_and_payoff(user, item_pool, i)
			self.knn_graph(i)
			#self.knn_signal()
			self.update_cluster_features(user)
			self.update_user_features(user, picked_item, payoff)
			self.find_regret(user, item_pool, payoff)

			if self.true_user_features is not None:
				error=np.linalg.norm(self.learned_user_features-self.true_user_features)
				self.learning_error.extend([error])
			else:
				pass 
			if self.true_graph is not None:
				error=np.linalg.norm(self.adj-self.true_graph)
				self.graph_error.extend([error])
			else:
				pass 
		return self.cum_regret,  self.adj, self.learned_user_features, self.learning_error,self.graph_error, self.denoised_signal
