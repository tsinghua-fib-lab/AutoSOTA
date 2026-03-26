
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, idim=512, odim=2, hdim=512, nlayers=2, dropout=0.1):
        super(MLP, self).__init__()
        self.idim = idim
        self.hdim = hdim
        self.nlayers = nlayers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.idim, self.hdim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.LayerNorm(self.hdim))
        self.layers.append(nn.Dropout(self.dropout))
        for i in range(self.nlayers-1):
            self.layers.append(nn.Linear(self.hdim, self.hdim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(self.hdim))
            self.layers.append(nn.Dropout(self.dropout))
        self.layers.append(nn.Linear(self.hdim, odim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def ce(logit, target):
    return (-target * F.log_softmax(logit, dim=-1)).sum()

def cross_entropy_2D(logit, target):
    return (ce(logit, target) + ce(logit.T, target.T))/2

def geodesic_mix(_lambda,a,b):
    theta = torch.acos((a*b).sum(dim=[1])).view(a.shape[0],1) + 1e-8
    n1 = torch.sin(_lambda*theta)/torch.sin(theta)*a
    n2 = torch.sin((1-_lambda)*theta)/torch.sin(theta)*b
    return n1+n2

class Aligner(nn.Module):
    def __init__(self, device, hdim=512, kg_dim=300, fp_dim=1024, ATC_adj=None,
                 alpha1 = 1, alpha2 = 1, alpha3 = 1):
        super(Aligner, self).__init__()
        self.device = device

        self.knowledge_encoder = nn.Sequential(
            nn.Linear(kg_dim, hdim),
            nn.LayerNorm(hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
        )
        self.structural_encoder = nn.Sequential(
            nn.Linear(fp_dim, hdim),
            nn.LayerNorm(hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
        )
        if ATC_adj is not None:
            self.ATC_adj = torch.FloatTensor(ATC_adj)

        # temperature parameters
        self.t1 = nn.parameter.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.t2 = nn.parameter.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def ce(logits,targets):
        return (-targets*nn.LogSoftmax(dim=-1)(logits)).sum()

    def forward(self, args, kg, structure, idx):
        kg = F.normalize(self.knowledge_encoder(kg))
        structure = F.normalize(self.structural_encoder(structure))

        I = torch.eye(kg.shape[0]).to(args.device)
        I.requires_grad = False
        ATC_I = self.ATC_adj[idx][:,idx].to(args.device)

        I_R = torch.flip(I, dims=[0])
        I_R.requires_grad = False
        I_X, I_XD = I+I_R, 1-(I+I_R)
        
        # # ATC_CLIP
        sim = torch.matmul(kg, structure.T)
        loss = cross_entropy_2D(sim/self.t1, ATC_I)
        
        # k-mix
        _lambda = torch.distributions.beta.Beta(self.alpha2, self.alpha2).sample().to(args.device)
        mix = geodesic_mix(_lambda, kg, kg.flip(dims=[0]))
        sim2 = torch.matmul(mix, structure.T)
        sim2 = sim*I_X + sim2*I_XD
        loss_m3 = cross_entropy_2D(sim2/self.t1, _lambda*I + (1-_lambda)*I_R)

        # s-mix
        _lambda = torch.distributions.beta.Beta(self.alpha2, self.alpha2).sample().to(args.device)
        mix = geodesic_mix(_lambda, structure, structure.flip(dims=[0]))
        sim_2_1 = torch.matmul(mix, kg.T)
        sim_2_1 = sim*I_X + sim_2_1*I_XD
        loss_m3_1 = cross_entropy_2D(sim_2_1/self.t1, _lambda*I + (1-_lambda)*I_R)

        # ks-mix
        _lambda = torch.distributions.beta.Beta(self.alpha3, self.alpha3).sample().to(args.device)
        mix_d = geodesic_mix(_lambda, kg, kg.flip(dims=[0]))
        mix_g = geodesic_mix(_lambda, structure, structure.flip(dims=[0]))
        sim2 = torch.matmul(mix_d, mix_g.T)
        sim2 = sim*I + sim2*(1-I)
        loss_m4 = cross_entropy_2D(sim2/self.t1, I)

        loss = 0.1*loss + (loss_m3 + loss_m3_1)/2 + loss_m4
        return loss
    
    def encode(self, kg, structure):
        kg = F.normalize(self.knowledge_encoder(kg))
        structure = F.normalize(self.structural_encoder(structure))

        return kg, structure

    def encode_fp(self, fp):
        from torch.utils import data as data_utils
        self.eval()
        dataset = data_utils.TensorDataset(fp)
        loader = data_utils.DataLoader(dataset, batch_size=1024, shuffle=False)
        encoded_fp = []
        with torch.no_grad():
            for data in loader:
                data = data[0].to(self.device)
                encoded_fp.append(self.structural_encoder(data).cpu())
        return torch.cat(encoded_fp, dim=0).numpy()

class BounDrE(nn.Module):
    '''
    EM-like optimization for drug one-class boundary with unlabeled compounds
    - encoder: MLP encodeer
    - nu: ratio of in-boundary drug samples
    - neg_lambda: loss weight for out-boundary compound samples
    - R: radius of the hypersphere
    - c: center of the hypersphere
    - decision_function: returns the distance from the center - if positive, inlier
    - score: returns the distance from the center
    '''
    def __init__(self, encoder, nu=0.95, neg_lambda=0.1):
        super(BounDrE, self).__init__()
        self.encoder = encoder # outputs odim
        self.nu = nu
        self.R = 0.0
        self.c = None
        self.neg_lambda = neg_lambda

    def get_c(self, loader):
        # calculate c without gradients
        self.eval()
        center = 0
        num_pos = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, label = data
                pos_inputs = inputs[label == 1]
                num_pos += len(pos_inputs)
                pos_inputs = pos_inputs.to(self.encoder.device)
                outputs = self.encoder(pos_inputs)
                center += outputs.sum(dim=0)
        center /= num_pos
        self.c = center
    
    def get_R(self, loader):
        # calculate R without gradients
        self.eval()
        with torch.no_grad():
            all_radius = []
            all_neg_radius = []
            for i, data in enumerate(loader):
                inputs, labels = data
                pos_inputs = inputs[labels == 1]
                neg_inputs = inputs[labels == 0]
                pos_inputs = pos_inputs.to(self.encoder.device)
                neg_inputs = neg_inputs.to(self.encoder.device)
                outputs = self.encoder(pos_inputs)
                neg_outputs = self.encoder(neg_inputs)
                radius = ((outputs - self.c)**2).sum(dim=1)
                neg_radius = ((neg_outputs - self.c)**2).sum(dim=1)
                all_radius.append(radius)
                all_neg_radius.append(neg_radius)
            R = torch.cat(all_radius)
            R = torch.quantile(R, self.nu)
            neg_R = torch.max(torch.cat(all_neg_radius))
            self.R = R
            self.R_comp = neg_R
        return R

    def decision_function(self, loader):
        # check if loader is not shuffled
        assert type(loader.batch_sampler.sampler) == torch.utils.data.sampler.SequentialSampler

        # calculate whether the data is inside the hypersphere; if positive, inlier
        self.eval()
        decision = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, _ = data
                inputs = inputs.to(self.encoder.device)
                outputs = self.encoder(inputs)
                radius = ((outputs - self.c)**2).sum(dim=1)
                decision.append((self.R - radius).cpu())   # if inside: positive, if outside: negative
        return torch.cat(decision) 

    def score(self, loader):
        # check if loader is not shuffled
        assert type(loader.batch_sampler.sampler) == torch.utils.data.sampler.SequentialSampler

        # calculate the distance of samples from the center
        self.eval()
        scores = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs, _ = data
                inputs = inputs.to(self.encoder.device)
                outputs = self.encoder(inputs)
                scores.append(((outputs - self.c)**2).sum(dim=1).cpu())
        scores = torch.cat(scores)/self.R
        return scores
    
    def score_from_embeddings(self, X_test):
        import torch.utils.data as data_utils
        dataset = data_utils.TensorDataset(torch.Tensor(X_test))
        loader = data_utils.DataLoader(dataset, batch_size=1024, shuffle=False)

        # calculate the distance of samples from the center
        self.eval()
        scores = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs = data[0]
                inputs = inputs.to(self.encoder.device)
                outputs = self.encoder(inputs)
                scores.append(((outputs - self.c)**2).sum(dim=1).cpu())
        scores = torch.cat(scores)/self.R.cpu()
        return scores

    def forward(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.encoder.device)
        labels = labels.to(self.encoder.device)
        
        outputs = self.encoder(inputs)
        drug_outputs = outputs[labels == 1]
        out_outputs = outputs[labels == 0]
        
        if len(drug_outputs) == 0:
            drug_loss = 0
        else:
            drug_loss = ((drug_outputs - self.c)**2).sum(dim=1).sum()

        if len(out_outputs) == 0:
            out_loss = 0
        else:
            out_loss =  torch.clamp(self.R_comp-((out_outputs - self.c)**2).sum(dim=1),min=0.0).sum()
        dist_loss = drug_loss + self.neg_lambda * out_loss 
        return dist_loss
    
