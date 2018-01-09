import tensorflow as tf
import numpy as np 
from scipy import misc
import matplotlib.pyplot as plt
from lap_pyramid import *
from tqdm import tqdm
import time

# Kernel taken from original laplace pyramid paper
kern = [[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
		  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
		  [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
		  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
		  [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]]
kern = np.reshape(kern, (5, 5, 1, 1))
#sigmas = np.arange(0.001, 5, 0.2)
sigmas = np.linspace(0.001, 0.1, 25)
# Reshape to use with tf.nn.conv2d
im1 = misc.imread('./car1.jpeg', 'L').astype(np.float32) / 255
img_r = im1.reshape(1, im1.shape[0], im1.shape[1], 1)
lap = lap_pyramid(6, im1.shape, kernel=kern)
start_time = time.time()
scores = []
eucl = []

for sig in tqdm(sigmas):
	noise = np.random.normal(0, sig, im1.shape)
	noisy_img = im1+noise
	noisy_img_r = noisy_img.reshape(1, im1.shape[0], im1.shape[1], 1)
	scores.append(lap.compare(img_r, noisy_img_r))
	eucl.append(np.sqrt(np.mean((im1-noisy_img)**2)))

print(time.time() - start_time)

plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(sigmas, scores, marker='x')
plt.title('Normalised Laplacian Pyramid')
plt.xlabel('Additive Gaussian Noise standard deviation')
plt.ylabel('Distance')
plt.subplot(2, 1, 2)
plt.scatter(sigmas, eucl, marker='x')
plt.title('Euclidian Distance')
plt.xlabel('Additive Gaussian Noise standard deviation')
plt.ylabel('Distance')
plt.show()
