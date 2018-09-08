import numpy as np 
import pandas as pandas
import os
os.chdir('C:/Kaige_Research/Graph Learning/graph_learning_code/')
from utils import *
from synthetic_data import *

class LINUCB_MAB():
	def __init__(self, user_num, item_num, dimension, item_pool_size, alpha, true_user_features=None, true_graph=None):
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=None
		self.dimension=dimension
		self.alpha=1+np.sqrt(np.log(2.0/alpha)/2.0)
		self.item_pool_size=item_pool_size
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
		self.learned_user_features=np.zeros((self.user_num, self.dimension))
		self.cum_regret=[0]
		self.learning_error=[]
		self.true_signal=None

	def pick_item_and_payoff(self, user, item_pool, time):
		mean=np.dot(self.item_features[item_pool], self.learned_user_features[user])
		temp1=np.dot(self.item_features[item_pool], np.linalg.inv(self.cov_matrix[user]))
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

	def update_user_features(self, user, picked_item, payoff):
		item_f=self.item_features[picked_item]
		self.cov_matrix[user]+=np.outer(item_f, item_f)
		self.bias[user]+=item_f*payoff
		self.learned_user_features[user]=np.dot(np.linalg.inv(self.cov_matrix[user]), self.bias[user])

	def find_regret(self, user, item_pool, payoff):
		max_payoff=np.max(self.noisy_signal[item_pool][:,user])
		regret=max_payoff-payoff
		self.cum_regret.extend([self.cum_regret[-1]+regret])


	def run(self,user_pool, item_pools, item_features, noisy_signal, true_signal, iteration):
		self.iteration=iteration
		self.noisy_signal=noisy_signal
		self.true_signal=true_signal
		self.item_features=item_features
		for i in range(self.iteration):
			print('LINUCB MAB Time ~~~~~~~~~~~~ ', i)
			user=user_pool[i]
			item_pool=item_pools[i]
			if user not in self.served_user:
				self.cov_matrix[user]=np.identity(self.dimension)
				self.bias[user]=np.zeros(self.dimension)
				self.picked_items_per_user[user]=[]
				self.payoffs_per_user[user]=[]
				self.served_user.extend([user])

			else:
				pass

			picked_item, payoff=self.pick_item_and_payoff(user, item_pool, i)
			self.update_user_features(user, picked_item, payoff)
			self.find_regret(user, item_pool, payoff)

			if self.true_user_features is not None:
				error=np.linalg.norm(self.learned_user_features-self.true_user_features)
				self.learning_error.extend([error])
			else:
				pass 

		return self.cum_regret, self.learned_user_features, self.learning_error