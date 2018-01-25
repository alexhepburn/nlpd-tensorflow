import numpy as np 
from lap_pyramid import *
import librosa
import os
import glob
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
import networkx as nx

from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx
from bokeh.models.sources import ColumnDataSource
from bokeh.models import Plot, MultiLine, Circle
from bokeh.models import GraphRenderer, Oval
from bokeh.palettes import Set3


import time

set3 = Set3[12]

def calc_rmse(l1, l2):
	rmse = [np.sqrt(np.mean((x - y) ** 2)) for (x, y) in zip(l1, l2)]
	return np.mean(rmse)

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
	lap = lap_pyramid(6, [2050, 513])
	df = create_pyramid_dataframe(flac_path, lap)
	scores = distance.cdist(list(df['Pyramid']), list(df['Pyramid']), metric=calc_rmse)

	# Create graph
	colors = [set3[x] for x in list(df['Album Number'])]
	plot = figure(title="LAPD Songs", x_range=(-1.1,1.1), y_range=(-1.1,1.1), 
		plot_width=800, plot_height=800)
	G = nx.from_numpy_matrix(scores)
	graph_renderer = from_networkx(G, nx.spring_layout)
	graph_renderer.node_renderer.data_source.add(colors, 'fillcolor')
	graph_renderer.node_renderer.glyph = Circle(size=15, fill_color='fillcolor')
	graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=2)
	plot.renderers.append(graph_renderer)
	output_file("song_graph.html")
	print("--- %s seconds ---" % (time.time()-start_time))
	show(plot)
