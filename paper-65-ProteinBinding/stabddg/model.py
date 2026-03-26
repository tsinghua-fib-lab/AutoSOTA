import torch
from torch import nn
from .mpnn_utils import featurize

class StaBddG(nn.Module):
    def __init__(self, pmpnn, use_antithetic_variates=True, noise_level=0.1, device='cuda'):
        super(StaBddG, self).__init__()
        self.pmpnn = pmpnn
        self.use_antithetic_variates = use_antithetic_variates
        self.noise_level = noise_level
        self.device = device

    def get_wt_seq(self, domain):
        """ Returns the wild type sequence of a protein. """
        _, wt_seq, *_ = featurize([domain], self.device)
        return wt_seq
    
    def folding_dG(self, domain, seqs, decoding_order=None, backbone_noise=None):
        """ Predicts the folding stability (dG) for a list of sequences. """
        B = seqs.shape[0]

        X_, _, mask_, _, chain_M_, residue_idx_, _, chain_encoding_all_ = featurize([domain], self.device)
        X_, S_, mask_ = X_.repeat(B, 1, 1, 1), seqs.to(self.device), mask_.repeat(B, 1)
        chain_M_ = chain_M_.repeat(B, 1)
        residue_idx_, chain_encoding_all_ = residue_idx_.repeat(B, 1), chain_encoding_all_.repeat(B, 1)
        
        order = decoding_order.repeat(B, 1) if self.use_antithetic_variates else None
        backbone_noise = backbone_noise.repeat(B, 1, 1, 1) if self.use_antithetic_variates else None

        log_probs = self.pmpnn(X_, S_, mask_, chain_M_, residue_idx_, chain_encoding_all_, 
                              fix_order=order, fix_backbone_noise=backbone_noise)

        seq_oh = torch.nn.functional.one_hot(seqs, 21).to(self.device)
        dG = torch.sum(seq_oh * log_probs, dim=(1, 2))

        return dG

    def folding_ddG(self, domain, mut_seqs, set_wt_seq=None):
        """ Predicts the folding ddG. """
        X, wt_seq, _, _, chain_M, _, _, _ = featurize([domain], self.device)

        if not set_wt_seq is None:
            wt_seq = set_wt_seq

        decoding_order = self._get_decoding_order(chain_M) if self.use_antithetic_variates else None
        backbone_noise = self._get_backbone_noise(X) if self.use_antithetic_variates else None
        
        wt_dG = self.folding_dG(domain, wt_seq, decoding_order=decoding_order, backbone_noise=backbone_noise)
        mut_dG = self.folding_dG(domain, mut_seqs, decoding_order=decoding_order, backbone_noise=backbone_noise)

        ddG = mut_dG - wt_dG

        return ddG
    
    def binding_ddG(self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        """ We calculate the binding ddG by decomposing it into three folding ddG terms, 
            corresponding to the entire complex and each individual binders. 
        """
        complex_ddG_fold = self.folding_ddG(complex, complex_mut_seqs)
        binder1_ddG_fold = self.folding_ddG(binder1, binder1_mut_seqs)
        binder2_ddG_fold = self.folding_ddG(binder2, binder2_mut_seqs)
        
        ddG = complex_ddG_fold - (binder1_ddG_fold + binder2_ddG_fold)
  
        return ddG

    def forward(self, complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs):
        return self.binding_ddG(complex, binder1, binder2, complex_mut_seqs, binder1_mut_seqs, binder2_mut_seqs)
    
    def _get_decoding_order(self, chain_M):
        """ Generate a random decoding order with the same shape as chain_M. """
        return torch.argsort(torch.abs(torch.randn(chain_M.shape, device=self.device)))
    
    def _get_backbone_noise(self, X):
        """ Generate random backbone noise. Defaults to 0.1A. """
        return self.noise_level * torch.randn_like(X, device=self.device)

class LinearModel(nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()
        # Linear layer with num_features inputs and 1 output (with bias term)
        self.linear = nn.Linear(num_features, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)
