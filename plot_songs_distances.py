import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn import manifold

def calc_rmse(l1, l2):
    rmse = [(np.sqrt(np.mean((x - y) ** 2))) for (x, y) in zip(l1, l2)]
    return np.mean(rmse)

df = pd.read_hdf('song_distances.h5', 'df')

scores = distance.cdist(list(df['Pyramid']), list(df['Pyramid']), metric=calc_rmse)
square_scores = distance.squareform(scores)

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(scores)
coords = results.embedding_
df['x'] = coords[:, 0]
df['y'] = coords[:, 1]

sns.lmplot('x', 'y', data=df, fit_reg=False, hue='Album Number')
plt.show()
