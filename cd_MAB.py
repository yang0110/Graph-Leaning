import numpy as np 
import pandas as pandas
import os
os.chdir('D:/Research/Graph Learning/code/')
from utils import *
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

class CD_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, K=10, true_user_features=None, true_graph=None):
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=None
		self.dimension=dimension
		self.alpha=alpha
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
		self.adj=None
		self.learned_cluster=range(self.user_num)
		self.learned_cluster_num=self.user_num
		self.learned_user_features=np.zeros((self.user_num, self.dimension))
		self.learned_cluster_features=np.zeros((self.user_num, self.dimension))
		self.adj=None
		self.lap=None
		self.cum_regret=[0]
		self.learning_error=[]
	def initial(self):
		for j in range(self.user_num):
			self.cov_matrix[j]=np.identity(self.dimension)
			self.bias[j]=np.zeros(self.dimension)
			self.cluster_cov_matrix[j]=np.identity(self.dimension)
			self.cluster_bias[j]=np.zeros(self.dimension)

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

	def find_cluster(self, user, time):

		if (time%1!=0):
			pass
		else:			

			self.adj=rbf_kernel(self.learned_user_features)
			rbf_row=self.adj[user].copy()
			small_index=np.argsort(rbf_row)[:self.user_num-self.K]
			rbf_row[small_index]=0.0
			self.adj[user,:]=rbf_row
			self.adj[:,user]=rbf_row

			graph=generate_graph_from_rbf(self.adj)
			self.learned_cluster, self.learned_cluster_num=find_community_best_partition(graph)
		print('CD cluster num ~~~~~~~~~~~~~~~~~~', self.learned_cluster_num)
		return self.learned_cluster, self.learned_cluster_num


	def update_cluster_features(self, user, time):
		if (time%1!=0):
			pass 
		else:
			same_cluster=np.where(np.array(self.learned_cluster)==self.learned_cluster[user])[0].tolist()
			sum_cov_matrix=np.identity(self.dimension)
			sum_bias=np.zeros(self.dimension)
			for key in same_cluster:
				sum_cov_matrix+=self.cov_matrix[key]-np.identity(self.dimension)
				sum_bias+=self.bias[key]
			self.cluster_cov_matrix[user]=np.identity(self.dimension)+sum_cov_matrix
			self.cluster_bias[user]=sum_bias
			inv_cluster_cor=np.linalg.inv(self.cluster_cov_matrix[user])
			new_cluster_feature=np.dot(inv_cluster_cor, self.cluster_bias[user])
			for i in same_cluster:
				self.learned_cluster_features[i]=new_cluster_feature


	def update_user_features(self, user, picked_item):
		signal=self.payoffs_per_user[user][-1]
		item_f=self.item_features[picked_item]
		self.cov_matrix[user]+=np.outer(item_f, item_f)
		self.bias[user]+=(item_f*signal).ravel()
		self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])

	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])

	def run(self, user_pool, item_pools, item_features, noisy_signal, iteration):
		self.iteration=iteration
		self.noisy_signal=noisy_signal
		self.item_features=item_features
		self.initial()
		for i in range(self.iteration):
			print('CD MAB Time ~~~~~~~~~~~~ ', i)
			user=user_pool[i]
			item_pool=item_pools[i]
			self.served_user=list(np.unique(user_pool[:i]))
			if user not in self.served_user:
				self.picked_items_per_user[user]=[]
				self.payoffs_per_user[user]=[]
			else:
				pass
			self.find_cluster(user, i)
			picked_item, payoff=self.pick_item_and_payoff(user, item_pool, i)
			self.update_user_features(user, picked_item)
			self.update_cluster_features(user, i)
			self.find_regret(user, item_pool, payoff)

			if self.true_user_features is not None:
				error=np.linalg.norm(self.learned_user_features-self.true_user_features)
				self.learning_error.extend([error])
			else:
				pass 

		return self.cum_regret, self.adj, self.learned_user_features, self.learning_error, self.learned_cluster