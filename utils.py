import numpy as np
from matplotlib.pylab import *
import argparse
import matplotlib.pyplot as plt
import networkx as nx 
import os 
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from synthetic_data import *


def sum_squareform(n):
	#sum operator that find degree from upper triangle
	ncols=int((n-1)*n/2)
	I=np.zeros(ncols)
	J=np.zeros(ncols)
	k=0
	for i in list((np.arange(n)+1))[1:]:

		I[k:k+(n-i)+1]=(np.arange(n)+1)[i-1:]
		k=k+(n-i+1)

	k=0
	for i in list((np.arange(n)+1))[1:]:
		J[k:k+(n-i)+1]=i-1
		k=k+n-i+1

	I=I.astype(int)
	J=J.astype(int)
	ys=list(I)+list(J)
	xs=list(np.arange(ncols)+1)+list(np.arange(ncols)+1)
	s_T=np.zeros((ncols,n))
	for i in range(len(ys)):
		s_T[(xs[i]-1),(ys[i]-1)]=1

	return s_T.T

def vector_form(W,n):
	w=W[np.triu_indices(n,1)]
	# the triangle-upper 
	return w


def lin_map(x, lims_out, lims_in):
	a=lims_in[0]
	b=lims_in[1]
	c=lims_out[0]
	d=lims_out[1]
	y=np.zeros(len([x]))
	y=((x-a)*(d-c)/(b-a))+c
	return y

