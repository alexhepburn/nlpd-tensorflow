import numpy as np
from lap_pyramid import *
import librosa
import os
import glob
import pandas as pd

filt2 = np.reshape([[0.0025, 0.0025, 0.0025, 0.0025, 0.0025],
                    [0.0025, 0.0025, 0.0105, 0.0025, 0.0025],
                    [0.1420, 0.2000, 0.2500, 0.2000, 0.1420],
                    [0.0025, 0.0025, 0.0105, 0.0025, 0.0025],
                    [0.0025, 0.0025, 0.0025, 0.0025, 0.0025]],
                    (5, 5, 1, 1)).astype(np.float32)

filt3 = np.reshape([[0.0045, 0.0045, 0.0045, 0.0045, 0.0045],
                    [0.0045, 0.0045, 0.0045, 0.0045, 0.0045],
                    [0.0045, 0.3003, 0.3003, 0.3003, 0.0045],
                    [0.0045, 0.0045, 0.0045, 0.0045, 0.0045],
                    [0.0045, 0.0045, 0.0045, 0.0045, 0.0045]],
                    (5, 5, 1, 1)).astype(np.float32)

def create_pyramid_dataframe(path, lap):
    # Batch process songs for STFT and Laplacian Pyramid Distance
    df = pd.DataFrame(columns=['Album', 'Song', 'Pyramid'])
    files = [item for sublist in [glob.glob(f+"*.flac")
        for f in (glob.glob(path + "*/"))] for item in sublist]
    raw_audio = np.asarray([librosa.load(x, duration=12.0)[0] for x in files])
    # Calculates STFT and resulting Pyramid
    pyramids = lap.output_pyramid_raw_audio(np.asarray(raw_audio))
    for i in range(0, len(files)):
        pyr = []
        for k in range(0, len(pyramids)):
            pyr.append(pyramids[k][i])
        splt = files[i].split(os.sep)
        df = df.append({'Song':splt[-1], 'Album':splt[-2], 'Pyramid':pyr}, ignore_index=True)
    album = pd.Series(df['Album']).astype('category')
    album_number = album.cat.codes
    df['Album Number'] = album_number
    return df

if __name__ == "__main__":
    start_time = time.time()
    flac_path = '/Users/ah13558/Documents/flac/'
    base_y, base_sr = librosa.load(flac_path+"base.flac", duration=12.0)
    lap = lap_pyramid(6, [2050, 513], kernel=filt2)
    df = create_pyramid_dataframe(flac_path, lap)
    df.to_hdf('song_distances.h5', 'df')
