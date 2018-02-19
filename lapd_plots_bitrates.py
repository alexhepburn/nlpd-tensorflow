import librosa
import numpy as np
import glob
from tqdm import tqdm
from mutagen.mp3 import MP3
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from lap_pyramid import *
from bokeh.palettes import Dark2
import scipy.io as sio
from bokeh.io import export_png
import pickle

def normalise(s):
    return (s - min(s)) / (max(s) - min(s))

pal = Dark2[8]
output_file('lapd_plot_bitrates.html')

audio_path = '/Users/ah13558/Documents/PhD/Image audio similarity/Audio Files/'
ODG = sio.loadmat('./ODG-bitrate.mat')
folders = (glob.glob(audio_path + "*/"))

bitrates = []
stft_eucl = []
stft_lap = []
chrom_eucl = []
chrom_lap = []

for fold in tqdm(folders):
    # Chromagrams from MATLAB
    chrom = sio.loadmat(fold + "/chrom.mat")['chroma']
    chromas = []
    names = []
    for i in range(0, chrom[0].shape[0]):
        try:
            names.append(float(chrom[0][i][0][0].replace('-', '.').replace('.mp3', '')))
            chromas.append(chrom[0][i][1][0][0][:, 0:514])
        except ValueError:
            y = chrom[0][i][1][0][0][:, 0:514]
    names_arr = np.asarray(names)
    x = np.argsort(names_arr)
    names = names_arr[x]
    chromas = [chromas[x1] for x1 in x]
    chrom_shape = chromas[0].shape
    lap_chrom = lap_pyramid(6, chrom_shape)
    yr = np.reshape(y, (1, chrom_shape[0], chrom_shape[1], 1))
    chrom_lap_temp, chrom_eucl_temp = [], []
    for c in chromas:
        cr = np.reshape(c, (1, chrom_shape[0], chrom_shape[1], 1))
        chrom_lap_temp.append(lap_chrom.compare(yr, cr))
        chrom_eucl_temp.append(np.sqrt(np.mean((y-c) ** 2)))

    for f in (glob.glob(fold + "*.wav")):
        y, sr = librosa.load(f, duration=12.0)
        c1s = librosa.core.stft(y=y)
        c1c = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=88)
        c1s = np.vstack((c1s.real, c1s.imag))
        stft_shape = c1s.shape
        c1sr = np.reshape(c1s, (1, stft_shape[0], stft_shape[1], 1))
        chrom_shape = c1c.shape
        c1cr = np.reshape(c1c, (1, chrom_shape[0], chrom_shape[1], 1))
        lap_stft = lap_pyramid(6, stft_shape)
        lap_chrom = lap_pyramid(6, chrom_shape)
    stft_lap_temp, stft_eucl_temp, bitrates = [], [], []
    for f in (glob.glob(fold + "*.mp3")):
        y2, sr2 = librosa.load(f, duration=12.0)
        m = MP3(f)
        bitrates.append(m.info.bitrate/1000)

        # STFT
        stft = librosa.core.stft(y=y2)
        stft = np.vstack((stft.real, stft.imag))
        stftr = np.reshape(stft, (1, stft_shape[0], stft_shape[1], 1))
        stft_lap_temp.append(lap_stft.compare(c1sr, stftr))
        stft_eucl_temp.append(abs(np.sqrt(np.mean((c1s - stft) ** 2))))

        # Chrom
        #chrom = librosa.feature.chroma_stft(y=y2, sr=sr2, n_chroma=88)
        #chromr = np.reshape(chrom, (1, chrom_shape[0], chrom_shape[1], 1))
        #chrom_lap_temp.append(lap_chrom.compare(c1cr, chromr))
        #chrom_eucl_temp.append(np.sqrt(np.mean((c1c - chrom) ** 2)))

    bit_array = np.asarray(bitrates)
    x = np.argsort(bit_array)
    bitrates = bit_array[x]
    stft_eucl.append(normalise(np.asarray(stft_eucl_temp)[x]))
    stft_lap.append(normalise(np.asarray(stft_lap_temp)[x]))
    chrom_lap.append(normalise(np.asarray(chrom_lap_temp)))
    chrom_eucl.append(normalise(np.asarray(chrom_eucl_temp)))

noise = {}
noise['stft_eucl'] = stft_eucl
noise['stft_lap'] = stft_lap
noise['chrom_lap'] = chrom_lap
noise['chrom_eucl'] = chrom_eucl
PEAQ = []
for i in range(0, len(folders)):
    PEAQ.append(np.abs(ODG['ODG'][i, :] - ODG['ODG'][i, 0]))
noise['PEAQ'] = PEAQ
noise['x'] = bitrates

with open('bitrates.pickle', 'wb') as handle:
    pickle.dump(noise, handle)

s1 = figure(width=500, plot_height=500, title='NLPD - STFT')
s1.xaxis.axis_label = 'Bitrate (k)'
for i in range(0, len(folders)):
    s1.line(bitrates, stft_lap[i], color=pal[i])

s2 = figure(width=500, plot_height=500, title='Euclidean - STFT')
s2.xaxis.axis_label = 'Bitrate (k)'
for i in range(0, len(folders)):
    s2.line(bitrates, stft_eucl[i], color=pal[i])

s3 = figure(width=500, plot_height=500, title='NLPD - Chromagram')
s3.xaxis.axis_label = 'Bitrate (k)'
for i in range(0, len(folders)):
    s3.line(bitrates, chrom_lap[i], color=pal[i])

s4 = figure(width=500, plot_height=500, title='Euclidean - Chromagram')
s4.xaxis.axis_label = 'Bitrate (k)'
for i in range(0, len(folders)):
    s4.line(bitrates, chrom_eucl[i], color=pal[i])

s5 = figure(width=500, plot_height=500, title= 'PEAQ')
s5.xaxis.axis_label = '`Biterate (k)'
for i in range(0, len(folders)):
    s5.line(bitrates, normalise(*abs(ODG['ODG'][i, :] - max(ODG['ODG'][i, :]))), color=pal[i])
p = gridplot([[s1, s2], [s3, s4], [None, s5]])

export_png(p, filename="lapd_bitrates.png")
