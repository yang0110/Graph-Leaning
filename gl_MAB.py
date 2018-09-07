import numpy as np 
import pandas as pandas
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from utils import *
from synthetic_data import *
from primal_dual_gl import Primal_dual_gl 
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

class GL_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, gl_alpha, gl_beta, gl_theta, gl_step_size=0.5,jump_step=10, mode=2, true_user_features=None, true_graph=None):
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
		self.learned_cluster_features=np.zeros((self.user_num, self.dimension))

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
		self.adj=np.identity(self.user_num)
		self.learning_error=[]
		self.served_user_num=None
		self.served_user=[]
		self.cum_regret=[0]
		self.graph_error=[]
		self.mode=mode
		self.jump_step=jump_step
		self.noisy_signal_copy=None
		self.mix_signal=None
		self.true_signal=None



	def pick_item_and_payoff(self, user, item_pool, time):

		mean=np.dot(self.item_features[item_pool], self.learned_cluster_features[user])
		temp1=np.dot(self.item_features[item_pool], np.linalg.inv(self.cov_matrix[user]))
		temp2=np.sum(temp1*self.item_features[item_pool], axis=1)*np.log(time+1)
		var=np.sqrt(temp2)
		pta=mean+self.alpha*var
		picked_item=item_pool[np.argmax(pta)]
		payoff=self.noisy_signal[picked_item, user]
		self.noisy_signal_per_user[user].extend([payoff])
		self.picked_items_per_user[user].extend([picked_item])
		if picked_item not in self.picked_items:
			self.picked_items.extend([picked_item])
		else:
			pass
		self.avaiable_noisy_signal=self.noisy_signal[self.picked_items]
		#print('self.avaiable_noisy_signal.shape', self.avaiable_noisy_signal.shape)
		self.mix_signal=self.noisy_signal_copy[self.picked_items]
		#print('self.mix_signal.shape', self.mix_signal.shape)
		return picked_item, payoff


	def graph_and_signal_learning(self, time):
		if (time%self.jump_step==0):
			if (self.mode==1) or (self.denoised_signal is None):
				print('mode 1, Update Graph On Noisy Signal')
				Z=euclidean_distances(self.avaiable_noisy_signal.T, squared=True)
			elif (self.mode==2):
				print('mode 2, Update Graph On Mixed Signal')
				Z=euclidean_distances(self.mix_signal.T, squared=True)
			elif (self.mode==3):
				print('mode 3, Update Graph On Denoised Signal')
				Z=euclidean_distances(self.denoised_signal.T, squared=True)
				#print('self.denoised_signal.shape', self.denoised_signal.shape)
			else:
				pass			

			np.fill_diagonal(Z,0)
			Z=norm_W(Z, self.user_num)
			primal_gl=Primal_dual_gl(self.user_num, Z, alpha=self.gl_alpha, beta=self.gl_beta, step_size=self.gl_step_size)
			primal_adj, error=primal_gl.run()
			self.adj=primal_adj.copy()
		else:
			pass

		lap=csgraph.laplacian(self.adj, normed=False)
		self.denoised_signal=np.dot(self.avaiable_noisy_signal, np.linalg.inv((np.identity(self.user_num)+self.gl_theta*lap)))
		self.noisy_signal_copy[self.picked_items]=self.denoised_signal

	def update_user_feature(self, user, picked_item):
		item_index=np.where(self.picked_items==picked_item)[-1]
		signal=self.denoised_signal[item_index, user]
		item_f=self.item_features[picked_item]
		self.cov_matrix[user]+=np.outer(item_f, item_f)
		self.bias[user]+=item_f*signal
		self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])

	def update_cluster_features(self, user):
		adj_copy=self.adj.copy()
		np.fill_diagonal(adj_copy,1)
		weights=adj_copy[user]
		self.learned_cluster_features[user]=np.average(self.learned_user_features,axis=0, weights=weights)


	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])


	def run(self, user_pool, item_pools, item_features, noisy_signal,true_signal, iteration):
		self.noisy_signal=noisy_signal.copy()
		self.noisy_signal_copy=noisy_signal.copy()
		self.true_signal=true_signal
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
			self.graph_and_signal_learning(i)
			self.update_cluster_features(user)
			self.update_user_feature(user, picked_item)
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
		return self.cum_regret,  self.adj, self.learned_user_features, self.learning_error, self.graph_error, self.denoised_signal



