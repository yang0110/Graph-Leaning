import os
os.chdir('D:/Research/Graph Learning/code/')
from pygsp import graphs, plotting, filters
import numpy as np
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import pyunlocbox
plotting.BACKEND='matplotlib'

a=0
# Create a random sensor graph
G = graphs.Sensor(N=256, distributed=True, seed=42)
G.compute_fourier_basis()
# Create label signal
label_signal = np.copysign(np.ones(G.N), G.U[:, 3])
G.plot_signal(label_signal)

rs = np.random.RandomState(42)
# Create the mask
M = rs.rand(G.N)
M = (M > 0.6).astype(float)  # Probability of having no label on a vertex.
# Applying the mask to the data
sigma = 0.1
subsampled_noisy_label_signal = M * (label_signal + sigma * rs.standard_normal(G.N))
G.plot_signal(subsampled_noisy_label_signal)

# Set the functions in the problem
gamma = 3.0
d = pyunlocbox.functions.dummy()
r = pyunlocbox.functions.norm_l1()
f = pyunlocbox.functions.norm_l2(w=M, y=subsampled_noisy_label_signal, lambda_=gamma)
# Define the solver
G.compute_differential_operator()
L = G.D.toarray()
step = 0.999 / (1 + np.linalg.norm(L))
solver = pyunlocbox.solvers.mlfbf(L=L, step=step)
# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob1 = pyunlocbox.solvers.solve([d, r, f], solver=solver, x0=x0, rtol=0, maxit=1000)

G.plot_signal(prob1['sol'])


# Set the functions in the problem
r = pyunlocbox.functions.norm_l2(A=L, tight=False)
# Define the solver
step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma * np.diag(M), 2)
solver = pyunlocbox.solvers.gradient_descent(step=step)
# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob2 = pyunlocbox.solvers.solve([r, f], solver=solver,
                                x0=x0, rtol=0, maxit=1000)

G.plot_signal(prob2['sol'])


import matplotlib.image as mpimg
import numpy as nptry
im_original = mpimg.imread('C:/Kaige_Research/Graph_based_recommendation_system/Result/preference_per_user.png')
im_original = np.dot(im_original[..., :3], [0.299, 0.587, 0.144])

np.random.seed(14)  # Reproducible results.
mask = np.random.uniform(size=im_original.shape)
mask = mask > 0.05

g = lambda x: mask * x
im_masked = g(im_original)
mask=1
g = lambda x: mask * x

from pyunlocbox import functions
f1 = functions.norm_tv(maxit=50, dim=2)

tau = 100
f2 = functions.norm_l2(y=im_masked, A=g, lambda_=tau)


from pyunlocbox import solvers
solver = solvers.forward_backward(step=0.5/tau)

x0 = np.array(im_masked)  # Make a copy to preserve im_masked.
ret = solvers.solve([f1, f2], x0, solver, maxit=100)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 2.5))
ax1 = fig.add_subplot(1, 3, 1)
_ = ax1.imshow(im_original, cmap='gray')
_ = ax1.axis('off')
_ = ax1.set_title('Original image')
ax2 = fig.add_subplot(1, 3, 2)
_ = ax2.imshow(im_masked, cmap='gray') 
_ = ax2.axis('off') 
_ = ax2.set_title('Masked image')
ax3 = fig.add_subplot(1, 3, 3)
_ = ax3.imshow(ret['sol'], cmap='gray')
_ = ax3.axis('off')
_ = ax3.set_title('Reconstructed image')

plt.show()