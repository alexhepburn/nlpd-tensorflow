from lap_pyramid import *
import numpy as np 
import matplotlib.pyplot as plt
import librosa, librosa.display
from scipy.spatial import distance
from tqdm import tqdm


y, sr = librosa.load(librosa.util.example_audio_file(), duration=10.0)
sigmas = np.linspace(0.01, 1, 25)
c1 = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=88)[:, 0:430]
chrom_shape = c1.shape
c1r = np.reshape(c1, (1, chrom_shape[0], chrom_shape[1], 1))

kern = [[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
		  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
		  [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
		  [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
		  [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]]
kern = np.reshape(kern, (5, 5, 1, 1))
lap = lap_pyramid(6, chrom_shape, kernel=kern)
scores = []
eucl = []
for sig in tqdm(sigmas):
	noise = np.random.normal(0, sig, y.shape[0])
	chrom = librosa.feature.chroma_cqt(y=(y+noise), sr=sr, n_chroma=88)[:, 0:430]
	chromr = np.reshape(chrom, (1, chrom_shape[0], chrom_shape[1], 1))
	scores.append(lap.compare(c1r, chromr))
	eucl.append(np.sqrt(np.mean((c1-chrom)**2)))

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

#librosa.output.write_wav('nonoise.wav', y, sr)
#librosa.output.write_wav('noise.wav', y+noise, sr)
#plt.figure()
#librosa.display.specshow(chromagram, y_axis='chroma')
#plt.show()