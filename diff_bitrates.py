from lap_pyramid import *
import librosa
from tqdm import *
import matplotlib.pyplot as plt
import glob
from mutagen.mp3 import MP3
from pydub import AudioSegment
from pydub.utils import mediainfo


files = (glob.glob("./Audio Files/*.mp3"))

y, sr = librosa.load('nonoise.wav', duration=10.0) # bitrate = 706k
#c1 = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=88)[:, 0:430]
c1 = librosa.core.stft(y=y)
chrom_shape = c1.shape
c1r = np.reshape(c1, (1, chrom_shape[0], chrom_shape[1], 1))

lap = lap_pyramid(6, chrom_shape)

scores = []
eucl = []
bitrates = []



for f in tqdm(files):
    y2, sr2 = librosa.load(f, duration=10.0)
    m = MP3(f)
    bitrates.append(m.info.bitrate/1000)
    #chrom = librosa.feature.chroma_cqt(y=y2, sr=sr2, n_chroma=88)[:, 0:430]
    chrom = librosa.core.stft(y=y2)
    chromr = np.reshape(chrom, (1, chrom_shape[0], chrom_shape[1], 1))
    scores.append(lap.compare(c1r, chromr))
    eucl.append(np.sqrt(np.mean((c1 - chrom) ** 2)))


print(eucl)
print(scores)
plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(bitrates, scores, marker='x')
plt.grid(True, which='both')
plt.title('Normalised Laplacian Pyramid')
plt.ylabel('Distance to original .wav')
plt.subplot(2, 1, 2)
plt.scatter(bitrates, eucl, marker='x')
plt.grid(True, which='both')
plt.title('Euclidian Distance')
plt.xlabel('Bitrate mp3 format')
plt.ylabel('Distance to original .wav')
plt.show()