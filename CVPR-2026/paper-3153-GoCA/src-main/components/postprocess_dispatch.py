import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import median_filter

from .postprocess.pamr import PAMR
# from .postprocess.dcrf import dcrf
from .postprocess.hungarian import hungarian_matching

def postprocess(task, data):
	if task == 'none':
		return data['cross_feat']
	elif task == 'affinity':
		return affinity_propagate(data['cross_feat'], data['space_feat'], data['order'])
	elif task == 'dcrf':
		return dcrf(data['cross_feat'], data['image_path'])
	elif task == 'merge_space':
		return space_merge(data['cross_feat'], data['space_feat'])
	elif task == 'rescaling':
		return rescaling(data['cross_feat'], data['target_token_id'], data['rescaling_token_id'])
	elif task == 'prototype':
		return prototype(data['cross_feat'], data['features'], data['dense_feat_id'], data['img_size'])
	elif task == 'hungarian prepare':
		return hungarian_prepare(data['cross_feat'], data['label'], data['class_count'])
	elif task == 'pamr':
		return pamr(data['cross_feat'], data['image'])


def affinity_propagate(cross_feat, space_feat, order):
	'''
	example:
	postprocess_setting = {
		'task': 'affinity',
		'order': 2,
	}
	'''
	null_supporting_target = cross_feat[-1].amin() == 0 and cross_feat[-1].amax() == 0
	if null_supporting_target:
		supporting_target = cross_feat[-1].unsqueeze(0)
		cross_feat = cross_feat[:-1,:,:]

	seq_len = cross_feat.shape[0]
	space_len = space_feat.shape[0]

	cross_feat = cross_feat.reshape(seq_len, space_len * space_len).permute(1, 0)
	space_feat = space_feat.reshape(space_len * space_len, space_len * space_len)

	for _ in range(order):
		cross_feat = space_feat @ cross_feat

	cross_feat = cross_feat.permute(1, 0).reshape(seq_len, space_len, space_len)
	amin = cross_feat.amin(dim=[1, 2], keepdim=True)
	amax = cross_feat.amax(dim=[1, 2], keepdim=True)
	cross_feat = (cross_feat - amin) / (amax - amin)

	if null_supporting_target:
		cross_feat = torch.concat([cross_feat, supporting_target], dim=0)

	return cross_feat


def space_merge(cross_feat, space_feat, threshold=2e-4):
	# part i: merge space attention
	kl_loss = torch.nn.KLDivLoss(reduction='none', log_target=True)
	def update_proposal(proposal, candidates):
		# proposal: space_len^2
		# candidates: b x space_len^2

		# calculate similarity
		proposal = proposal.unsqueeze(0)
		proposal_log = (proposal + 1e-4).log()
		candidates_log = (candidates + 1e-4).log()
		similarity = kl_loss(proposal_log, candidates_log) + kl_loss(candidates_log, proposal_log)
		# print(similarity.shape)  # b x space_len^2
		similarity = similarity.mean(dim=1)
		# print(similarity.shape)  # b

		# now we have the similarity between proposal and each candidate
		# find those more similar than the given threshold and average them
		mask = (similarity < threshold).unsqueeze(1)
		updated_proposal = (candidates * mask).sum(dim=0) / mask.sum()
		# normalization
		updated_proposal = updated_proposal / updated_proposal.sum()

		return updated_proposal, mask.squeeze(1)

	# generate proposals by uniformly sampling from the map
	space_len = space_feat.shape[0]
	grid_len = 4
	grid_num = space_len // grid_len
	proposals = []
	for i in range(grid_num):
		for j in range(grid_num):
			pixel_i = i * grid_len
			pixel_j = j * grid_len
			proposals.append(space_feat[pixel_i, pixel_j].reshape(-1))

	# initialize proposals by averaging similar maps
	space_feat = space_feat.reshape(space_len * space_len, space_len * space_len)
	new_proposals = []
	for proposal in proposals:
		new_proposals.append(update_proposal(proposal, space_feat)[0])
	proposals = torch.stack(new_proposals)

	# merge proposals
	for _ in range(3):
		# proposals: b x space_len
		new_proposals = []
		while True:
			# grab a proposal to merge
			proposal = proposals[0]

			# update proposal
			updated_proposal, mask = update_proposal(proposal, proposals)
			new_proposals.append(updated_proposal)

			# exclude the averaged proposals and update the remaining ones
			mask = mask == False
			if mask.sum() == 0:
				break
			proposals = proposals[mask.nonzero(as_tuple=True)]
		proposals = torch.stack(new_proposals)


	# part ii: assign labels to proposals
	cross_feat = cross_feat.reshape(-1, space_len * space_len)
	# proposals: b x space_len^2
	# cross_feat: n x space_len^2

	if True:
		# convert proposals into hard mask
		hard_proposals = proposals.argmax(dim=0)
		proposals = torch.zeros(proposals.shape).to(proposals.device, proposals.dtype)
		for idx in range(proposals.shape[0]):
			proposals[idx] = (hard_proposals == idx) * 1

		# calculate token weights with regard to each proposal
		proposal_to_token = proposals @ cross_feat.permute(1, 0)  # b x n, this equals sum
		proposal_to_token = proposal_to_token / proposals.sum(dim=1, keepdim=True)  # get mean from sum

		# merge hard proposals into each token's hard mask
		cross_feat = torch.zeros(cross_feat.shape).to(cross_feat.device, cross_feat.dtype)
		for proposal_idx in range(proposal_to_token.shape[0]):
			value, token_idx = proposal_to_token[proposal_idx].max(dim=0)
			if value > 0.5:
				cross_feat[token_idx] = cross_feat[token_idx] + proposals[proposal_idx]
		# there can be pixel without corresponding token, and token without corresponding pixel
		# which is valid
	else:
		affinity = proposals.permute(1, 0) @ proposals
		for _ in range(3):
			cross_feat = cross_feat @ affinity

	cross_feat = cross_feat.reshape(-1, space_len, space_len)
	return cross_feat


# rescale_method = 'raw'
rescale_method = 'per-token renorm'
# rescale_method = 'sum-1 rescaling'
# rescale_method = 'sum-1 rescaling + per-token renorm'
# rescale_method = 'sum-1 rescaling + per-token renorm+'
# rescale_method = 'sum-1 rescaling + per-token renorm x raw'
# rescale_method = 'sum-1 rescaling + per-token renorm+ x raw'
def rescaling(aggregated_attn, target_token_id, rescaling_token_id):
	raw_scores = aggregated_attn.permute(1, 2, 0)  # space x space x n

	if rescale_method == 'raw':
		processed_scores = raw_scores
	elif rescale_method == 'per-token renorm':
		processed_scores = raw_scores / raw_scores.amax(dim=[0, 1], keepdim=True)
	elif rescale_method == 'sum-1 rescaling':
		# processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
		factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
		processed_scores = raw_scores / factor
	elif rescale_method == 'sum-1 rescaling + per-token renorm':
		# processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
		factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
		processed_scores = raw_scores / factor
		processed_scores = processed_scores / processed_scores.amax(dim=[0, 1], keepdim=True)
	elif rescale_method == 'sum-1 rescaling + per-token renorm+':
		# processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
		factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
		processed_scores = raw_scores / factor
		processed_scores = torchvision.transforms.functional.gaussian_blur(
			processed_scores.permute(2, 0, 1), kernel_size=5
		).permute(1, 2, 0)
		amin = processed_scores.amin(dim=[0, 1], keepdim=True)
		amax = processed_scores.amax(dim=[0, 1], keepdim=True)
		processed_scores = (processed_scores - amin) / (amax - amin)
	elif rescale_method == 'sum-1 rescaling + per-token renorm x raw':
		# processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
		factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
		processed_scores = raw_scores / factor
		processed_scores = processed_scores / processed_scores.amax(dim=[0, 1], keepdim=True)
		processed_scores = processed_scores * raw_scores
	elif rescale_method == 'sum-1 rescaling + per-token renorm+ x raw':
		# processed_scores = raw_scores / (1 - raw_scores[:,:,0]).unsqueeze(-1)
		factor = raw_scores[:,:,rescaling_token_id].sum(dim=2, keepdim=True)
		processed_scores = raw_scores / factor
		processed_scores = torchvision.transforms.functional.gaussian_blur(
			processed_scores.permute(2, 0, 1), kernel_size=5
		).permute(1, 2, 0)
		amin = processed_scores.amin(dim=[0, 1], keepdim=True)
		amax = processed_scores.amax(dim=[0, 1], keepdim=True)
		processed_scores = (processed_scores - amin) / (amax - amin)
		processed_scores = processed_scores * raw_scores

	all_class_feat = processed_scores[:,:,target_token_id].permute(2, 0, 1)
	return all_class_feat


def prototype(attn, features, dense_id, target_size):
	dense = features[dense_id]
	prototypes = []

	# generate prototypes
	for idx in range(attn.shape[0]):
		class_mask = attn[idx]
		# print(class_mask.shape, dense.shape)  # h x w // dim x h' x w'
		class_mask = torch.nn.functional.interpolate(
			class_mask.unsqueeze(0).unsqueeze(0), (dense.shape[1], dense.shape[2]),
			mode='nearest'
		).squeeze(0)
		p = (class_mask * dense).sum(dim=[1, 2]) / (class_mask.sum() + 1e-6)  # dim
		prototypes.append(p)

	# resize dense feature to image size
	dense = torch.nn.functional.interpolate(
		dense.unsqueeze(0), target_size,
		mode='bicubic'
	).squeeze(0)

	# calculate cosine similarity between prototypes and per-pixel features
	dim, h, w = dense.shape
	dense = dense.reshape(dim, -1).permute(1, 0)  # (h x w) x dim
	scores = []
	for p in prototypes:
		score = torch.nn.functional.cosine_similarity(p.unsqueeze(0), dense, dim=1).reshape(h, w)
		# print(scores.shape)  # h x w
		scores.append(score)
	scores = torch.stack(scores)  # n x h x w

	# assign each pixel to a class based on argmax
	assignment = scores.argmax(dim=0)  # h x w
	output = torch.zeros([scores.shape[0], h, w], dtype=dense.dtype).to(dense.device)  # n x h x w
	for idx in range(output.shape[0]):
		output[idx] = output[idx] + 1 * (assignment == idx)
	return output

	# re-normalization
	# scores = scores / (scores.sum(dim=0, keepdim=True) + 1e-6)
	# amin = scores.amin(dim=[1, 2], keepdim=True)
	# amax = scores.amax(dim=[1, 2], keepdim=True)
	# scores = (scores - amin) / (amax - amin)
	# return scores


# https://github.com/PaulCouairon/DiffCut/blob/106635809747b20cf5a3aefe5c79a01ee20575bb/eval_diffcut.py
def hungarian_prepare(cross_feat, label, n_class):
	# cross_feat: n x h x w
	# label: h x w
	cross_feat = F.interpolate(cross_feat.unsqueeze(0), label.shape, mode='bilinear').squeeze(0)
	pred = cross_feat.argmax(dim=0)  # h x w

	label = label - 1
	_, _, _, _, hist, col_ind = hungarian_matching(pred.detach().cpu(), label, n_class-1)
	label[label==-1] = n_class - 1
	# Assign a unique label to the background in pred maps
	assigned_gt_clusters = np.where(hist.max(axis=1)>0)[0].tolist()
	assigned_pred_clusters = [col_ind[i] for i in assigned_gt_clusters]
	background_clusters = list(set(np.unique(pred.detach().cpu().numpy())) - set(assigned_pred_clusters))
	bg_label = pred.detach().cpu().numpy().max() + 1
	for bg_cls in background_clusters:
		pred[pred==bg_cls] = bg_label

	return pred


def pamr(pred, image):
	image = torch.from_numpy(np.array(image)).to(pred.device).float()
	image = image.permute(2, 0, 1)
	im =  F.interpolate(image.unsqueeze(0), size=pred.shape, mode='bilinear')

	def pamr(labels, image):
		masks = torch.cat([1. * (labels == label) for label in torch.unique(labels)], dim=1)
		masks = masks.unsqueeze(0).unsqueeze(0)
		labels = PAMR(num_iter=10, dilations=[1, 2, 4, 8])(image, masks) # 1, 2, 4, 8
		labels = 1. * torch.argmax(labels, dim=1)
		labels = median_filter(labels.cpu().numpy(), 3).astype(int)
		return labels

	pred = torch.from_numpy(pamr(pred, im)).to(pred.device).squeeze(0)
	return pred
