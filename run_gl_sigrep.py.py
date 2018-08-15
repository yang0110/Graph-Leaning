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
from community import community_louvain
import pandas as pd 
import csv
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from collections import Counter
from scipy.sparse import csgraph
import seaborn as sns
from synthetic_data import *
from gl_sigrep import Gl_sigrep 

node_num=20
signal_num=100
rbf_adj, rbf_lap=rbf_graph(node_num)
signals=generate_signal(signal_num, node_num, rbf_lap)

iteration=300
alpha=0.012
beta=0.79

gl_sigrep=Gl_sigrep(node_num, signals, iteration, alpha, beta)
lap, output=gl_sigrep.run()

error=np.linalg.norm(output-signals)/signal_num

lap_error=np.linalg.norm(rbf_lap-lap, 'fro')/signal_num

fig, (ax1, ax2)=plt.subplots(1,2, figsize=(8,4))
ax1.imshow(rbf_lap)
ax1.set_title('Ground Truth Laplacian')
ax2.imshow(lap)
ax2.set_title('Learned Laplacian')
plt.show()