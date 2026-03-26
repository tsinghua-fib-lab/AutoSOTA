import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable

class hetero_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(hetero_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat1, feat2, label1):
		feat_size = feat1.size()[1]
		feat_num = feat1.size()[0]
		label_num =  len(label1.unique())
		feat1 = feat1.chunk(label_num, 0)
		feat2 = feat2.chunk(label_num, 0)
		#loss = Variable(.cuda())
		for i in range(label_num):
			center1 = torch.mean(feat1[i], dim=0)
			center2 = torch.mean(feat2[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, abs(self.dist(center1, center2)))
				else:
					dist += max(0, abs(self.dist(center1, center2)))
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center2))
				else:
					dist += max(0, 1-self.dist(center1, center2))

		return dist
		