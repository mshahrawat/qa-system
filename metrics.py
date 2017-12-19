import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

def cos_sim(q, others):
	cos_sims = F.cosine_similarity(q.unsqueeze(0).expand_as(others), others)
	_, sorted_indices = torch.sort(cos_sims, dim=0, descending=True)
	return cos_sims, sorted_indices

def mrr(q, others, labels):
	cos_sims, sorted_indices = cos_sim(q, others)
	for i in xrange(len(sorted_indices)):
		idx = sorted_indices[i]
		if labels[idx] == 1:
			return 1. / (i + 1)
	return 0

def p_at_n(n, q, others, labels):
	cos_sims, sorted_indices = cos_sim(q, others)
	pos_qs = 0.	
	for i in xrange(n):
		idx = sorted_indices[i]
		if labels[idx] == 1:
			pos_qs += 1
	return pos_qs / n

def map(q, others, labels, total_pos):
	if total_pos == 0:
		return 0
	num_pos = 0.
	prec_sum = 0.
	cos_sims, sorted_indices = cos_sim(q, others)
	for k in xrange(len(sorted_indices)):
		idx = sorted_indices[k]
		if labels[idx] == 1:
			num_pos += 1
			current_prec = num_pos / float(k + 1)
		else:
			current_prec = 0
		prec_sum += current_prec
		if num_pos == total_pos:
			break
	return prec_sum / float(num_pos)

q = torch.rand(667)
others = torch.rand(21, 667)
labels = torch.zeros(21)
cos, ranks = cos_sim(q, others)