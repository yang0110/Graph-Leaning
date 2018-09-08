import numpy as np 
import pandas as pandas
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from utils import *
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.metrics import normalized_mutual_info_score
class CD_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, K=10, jump_step=10,true_user_features=None, true_graph=None):
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
		self.picked_items_per_user={}
		self.payoffs_per_user={}
		self.picked_items=[]
		self.iteration=None
		self.cov_matrix={}
		self.bias={}
		self.cluster_cov_matrix={}
		self.cluster_bias={}
		self.learned_cluster=range(self.user_num)
		self.learned_cluster_num=self.user_num
		self.learned_user_features=np.zeros((self.user_num, self.dimension))
		self.learned_cluster_features=np.zeros((self.user_num, self.dimension))
		self.adj=np.identity(self.user_num)
		self.lap=None
		self.cum_regret=[0]
		self.learning_error=[]
		self.graph_error=[]
		self.cluster_error=[]
		self.true_label=None
		self.jump_step=jump_step
		self.not_served_user=list(range(self.user_num))
		self.true_signal=None
		self.avaiable_noisy_signal=None

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
		if (time%self.jump_step!=0):
			pass
		else:			
			print('Update Graph')
			if self.avaiable_noisy_signal is None:
				self.adj=rbf_kernel(self.learned_cluster_features)
			else:
				self.adj=rbf_kernel(self.avaiable_noisy_signal.T)
			rbf_row=self.adj[user].copy()
			if len(self.not_served_user)==0:
				pass 
			else:
				rbf_row[self.not_served_user]=0
			small_index=np.argsort(rbf_row)[:self.user_num-self.K]
			#big_index=np.argsort(rbf_row)[self.user_num-self.K:]
			rbf_row[small_index]=0.0
			#rbf_row[big_index]=1.0
			self.adj[user,:]=rbf_row
			self.adj[:,user]=rbf_row
			graph=generate_graph_from_rbf(self.adj)
			self.learned_cluster, self.learned_cluster_num=find_community_best_partition(graph)
			error=normalized_mutual_info_score(self.learned_cluster, self.true_label)
			self.cluster_error.extend([error])
		print('CD cluster num ~~~~~~~~~~~~~~~~~~~~~~', self.learned_cluster_num)
		return self.learned_cluster, self.learned_cluster_num


	def update_cluster_features(self, user, time):
		print('Update Cluster Features')
		same_cluster=np.where(np.array(self.learned_cluster)==self.learned_cluster[user])[0].tolist()
		sum_cov_matrix=np.identity(self.dimension)
		sum_bias=np.zeros(self.dimension)
		for key in same_cluster:
			sum_cov_matrix+=(self.cov_matrix[key]-np.identity(self.dimension))
			sum_bias+=self.bias[key]

		self.cluster_cov_matrix[user]=sum_cov_matrix
		self.cluster_bias[user]=sum_bias
		inv_cluster_cor=np.linalg.inv(self.cluster_cov_matrix[user])
		new_cluster_feature=np.dot(inv_cluster_cor, self.cluster_bias[user])
		#weights=self.adj[user][same_cluster]
		#new_cluster_feature=np.average(self.learned_user_features[same_cluster], axis=0, weights=weights)
		for i in same_cluster:
			self.learned_cluster_features[i]=new_cluster_feature


	def update_user_features(self, user, picked_item, payoff):
		item_f=self.item_features[picked_item]
		self.cov_matrix[user]+=np.outer(item_f, item_f)
		self.bias[user]+=item_f*payoff
		self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])

	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])

	def run(self, user_pool, item_pools, item_features, noisy_signal, true_signal, iteration, true_label):
		self.iteration=iteration
		self.noisy_signal=noisy_signal
		self.true_signal=true_signal
		self.item_features=item_features
		self.true_label=true_label
		self.initial()
		for i in range(self.iteration):
			print('CD MAB Time ~~~~~~~~~~~~ ', i)
			user=user_pool[i]
			item_pool=item_pools[i]
			if user not in self.served_user:
				self.picked_items_per_user[user]=[]
				self.payoffs_per_user[user]=[]
				self.served_user.extend([user])
				self.not_served_user.remove(user)

			else:
				pass

			self.find_cluster(user, i)
			picked_item, payoff=self.pick_item_and_payoff(user, item_pool, i)
			self.update_user_features(user, picked_item, payoff)
			self.update_cluster_features(user, i)
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
		return self.cum_regret, self.adj, self.learned_user_features, self.learned_cluster_features, self.learning_error, self.graph_error, self.learned_cluster, self.cluster_error
