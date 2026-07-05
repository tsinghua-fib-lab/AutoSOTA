import torch
import torch.nn.functional as F
import torch.nn as nn

class Gprompt_tuning_loss(nn.Module):
    def __init__(self, tau=0.1):
        super(Gprompt_tuning_loss, self).__init__()
        self.tau = tau
    
    def forward(self, embedding, center_embedding, labels):
        # For each sample pair (xi,yi), the loss is -ln(+sim / +sim + -sim).

        # Calculate the similarity between all samples and all class prototypes
        similarity_matrix = F.cosine_similarity(embedding.unsqueeze(1), center_embedding.unsqueeze(0), dim=-1) / self.tau
        exp_similarities = torch.exp(similarity_matrix)
        # Sum exponentiated similarities for the denominator
        pos_neg = torch.sum(exp_similarities, dim=1, keepdim=True)
        # select the exponentiated similarities for the correct classes for the every pair (xi,yi)
        pos = exp_similarities.gather(1, labels.view(-1, 1))
        L_prompt = -torch.log(pos / pos_neg)
        loss = torch.sum(L_prompt)
                    
        return loss

def Gprompt_link_loss(node_emb, pos_emb, neg_emb, temperature=0.2):
    r"""Refer to GraphPrompt original codes"""
    x = torch.exp(F.cosine_similarity(node_emb, pos_emb, dim=-1) / temperature)
    y = torch.exp(F.cosine_similarity(node_emb, neg_emb, dim=-1) / temperature)

    loss = -1 * torch.log(x / (x + y) )
    return loss.mean()