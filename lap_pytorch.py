import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class lap_pyramid():

	def __init__(self, k, image_size, filt=np.reshape([[0.0025, 0.0125, 0.0200, 0.0125, 0.0025],
                   									   [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                   									   [0.0200, 0.1000, 0.1600, 0.1000, 0.0200],
                   									   [0.0125, 0.0625, 0.1000, 0.0625, 0.0125],
                   									   [0.0025, 0.0125, 0.0200, 0.0125, 0.0025]]*3,
                   									   (3, 1, 5, 5)).astype(np.float32)):
		self.k = k
		self.filt = torch.Tensor(filt)
		self.image_size = image_size
		self.dn_filts, self.sigmas = self.DN_filters()
		self.pad_one = nn.ReflectionPad2d(1)
		self.pad_two = nn.ReflectionPad2d(2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

	def DN_filters(self):
		sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
		dn_filts = []
		dn_filts.append(torch.Tensor(np.reshape([[0, 0.1011, 0],
		                            [0.1493, 0, 0.1460],
		                            [0, 0.1015, 0.]]*3,
		                           (3,	1, 3, 3)).astype(np.float32)))

		dn_filts.append(torch.Tensor(np.reshape([[0, 0.0757, 0],
		                            [0.1986, 0, 0.1846],
		                            [0, 0.0837, 0]]*3,
		                           (3, 1, 3, 3)).astype(np.float32)))

		dn_filts.append(torch.Tensor(np.reshape([[0, 0.0477, 0],
		                            [0.2138, 0, 0.2243],
		                            [0, 0.0467, 0]]*3,
		                           (3, 1, 3, 3)).astype(np.float32)))

		dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
		                            [0.2503, 0, 0.2616],
		                            [0, 0, 0]]*3,
		                           (3, 1, 3, 3)).astype(np.float32)))

		dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
		                            [0.2598, 0, 0.2552],
		                            [0, 0, 0]]*3,
		                           (3, 1, 3, 3)).astype(np.float32)))

		dn_filts.append(torch.Tensor(np.reshape([[0, 0, 0],
		                            [0.2215, 0, 0.0717],
		                            [0, 0, 0]]*3,
		                           (3, 1, 3, 3)).astype(np.float32)))

		return dn_filts, sigmas

	def normalise(self, convs):
		norm = []
		for i in range(0, len(convs)):
			n = F.conv2d(self.pad_one(torch.abs(convs[i])), self.dn_filts[i], stride=1, groups=3)
			norm.append(convs[i]/(self.sigmas[i]+n))
		return norm

	def pyramid(self, im):
		out = []
		J = im 
		pyr = []
		for i in range(0, self.k):
			print(J.shape)
			I = F.conv2d(self.pad_two(J), self.filt, stride=2, padding=0, groups=3)
			I_up = self.upsample(I)
			I_up_conv = F.conv2d(self.pad_two(I_up), self.filt, stride=1, padding=0, groups=3)
			pyr.append(J - I_up_conv)
			J = I
		return self.normalise(pyr)
