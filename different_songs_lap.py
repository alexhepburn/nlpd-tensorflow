import numpy as np 
from lap_pyramid import *
import librosa
import os
import glob
import pandas as pd

df = pd.DataFrame(columns=['Album', 'Song', 'Pyramid'])

flac_path = audio_path = '/Users/ah13558/Documents/flac/'
base_y, base_sr = librosa.load(flac_path+"base.flac", duration=12.0)
folders = (glob.glob(audio_path + "*/"))
base_stft = librosa.core.stft(y=base_y)
base_stft = np.vstack((base_stft.real, base_stft.imag))
stft_shape = base_stft.shape
base_stft = np.reshape(base_stft, (1, stft_shape[0], stft_shape[1], 1))
lap = lap_pyramid(6, stft_shape)

def stft_reshape(f):
	y, sr = librosa.load(files[i], duration=12.0)
	stft = librosa.core.stft(y=y)
	stft = np.vstack((stft.real, stft.imag))
	stft = np.reshape(stft, (stft_shape[0], stft_shape[1], 1))

def calc_rmse(l1, l2):
	rmse = []
	for i in range(0, self.k):
            rmse.append(np.sqrt(np.mean((convs_up_out1[i] - convs_up_out2[i]) ** 2)))
    return mean(rmse)

pyramid_dict = {}
for fold in folders[0:1]:
	temp_dict = {}
	multiple_stft = []
	files = glob.glob(fold + "*.flac")
	multiple_stft = np.ones((len(files), stft_shape[0], stft_shape[1], 1))
	# Prepare files for batch processing to get the laplacian pyramids for each
	# song.
	for i in range(0, len(files)):
		y, sr = librosa.load(files[i], duration=12.0)
		stft = librosa.core.stft(y=y)
		stft = np.vstack((stft.real, stft.imag))
		stft = np.reshape(stft, (stft_shape[0], stft_shape[1], 1))
		multiple_stft[i, :, :, :] = stft
	# Computer laplacian pyramid for batch
	pyramids = lap.output_pyramid(multiple_stft)
	# Sort batch into a album of dictionaries which contain a dictionary for songs
	for i in range(0, len(files)):
		pyr = []
		# Collect all pyramids in a list for each song
		for k in range(0, len(pyramids)):
			pyr.append(pyramids[k][i])
		# Store pyramids as values and song names as keys
		df['Song'] = files[i].split(os.sep)[-1]
		df['Album'] = fold.split(os.sep)[-2]
		df['Pyramid'] = pyr
		print(fold.split(os.sep)[-2] + " DONE: " + str(len(files)) + " FILES FOUND")

print(df)
