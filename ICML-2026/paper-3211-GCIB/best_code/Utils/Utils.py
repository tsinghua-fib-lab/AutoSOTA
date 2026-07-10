import torch
import numpy as np
import inspect
import torch.nn.functional as F


def innerProduct(usrEmbeds, itmEmbeds):
	return torch.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


# 因为使用了全体的嵌入 所以不是batch内的正则
def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2) / W.shape[0]
		# ret += W.norm(2).square()
		# ret += W.norm(2).square() / W.shape[0]
	return ret


def InfoNce(view1, view2, temperature: float):
	# embedding做归一化
	view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
	score = (view1 @ view2.T) / temperature
	# 做log_softmax(dim=1)  取对角线的值 平均 给负
	score = -torch.diag(F.log_softmax(score, dim=1)).mean()
	return score







