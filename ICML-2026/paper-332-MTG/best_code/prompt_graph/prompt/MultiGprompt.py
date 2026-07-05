import torch
import torch.nn as nn
import torch.nn.functional as F


class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3, a4, ft_in, nb_classes, device):
        super(downprompt, self).__init__()
        self.downprompt = downstreamprompt(ft_in)
        self.nb_classes = nb_classes
        self.a4 = a4
        self.leakyrelu = nn.ELU()
        self.device = device
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.nodelabelprompt = weighted_prompt(3)
        self.dffprompt = weighted_feature(2)
        self.aveemb0 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb1 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb2 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb3 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb4 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb5 = torch.FloatTensor(ft_in, ).to(self.device)
        self.aveemb6 = torch.FloatTensor(ft_in, ).to(self.device)

        self.one = torch.ones(1, ft_in).to(self.device)
        self.ave = torch.FloatTensor(nb_classes, ft_in).to(self.device)

    def forward(self, seq, seq1, labels, train=0):

        weight = self.leakyrelu(self.nodelabelprompt(self.prompt))
        weight = self.one + weight
        rawret1 = weight * seq
        rawret2 = self.downprompt(seq)
        rawret4 = seq1
        rawret3 = self.dffprompt(rawret1, rawret2)
        rawret = rawret3 + self.a4 * rawret4
        rawret = rawret.to(self.device)
        if train == 1:
            self.ave = averageemb(labels, rawret, self.nb_classes).to(self.device)

        ret = torch.FloatTensor(seq.shape[0], self.nb_classes).to(self.device)
        for x in range(seq.shape[0]):
            for i in range(self.nb_classes):
                ret[x][i] = torch.cosine_similarity(rawret[x], self.ave[i], dim=0)

        ret = F.softmax(ret, dim=1)

        # ret = torch.argmax(ret, dim=1)
        # print('ret=', ret)

        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


# def averageemb(labels, rawret, nb_class):
#     retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1]))

#     counters = {i: 0 for i in range(nb_class)}

#     for x in range(rawret.shape[0]):
#         label = labels[x].item()
#         if label < nb_class:
#             retlabel[label][counters[label]] = rawret[x]
#             counters[label] += 1

#     retlabel = torch.mean(retlabel, dim=1)

#     return retlabel


# ours
def averageemb(index, input, label_num):
    device = input.device
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if (class_counts[i].item() == 0):
            continue
        else:
            c[i] /= class_counts[i]

    return c


class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.5)
        self.weight[0][1].data.fill_(0.4)
        self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        # print("weight",self.weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding


class featureprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)

    def forward(self, feature):
        # print("prompt",self.weightprompt.weight)
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act=None, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, input, sparse=True):
        # print("input",input)
        seq = input[0]
        adj = input[1]
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.mm(adj.squeeze(dim=0), seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout):
        super(GcnLayers, self).__init__()

        self.act = torch.nn.ReLU()
        self.num_layers_num = num_layers_num
        self.g_net, self.bns = self.create_net(n_in, n_h, self.num_layers_num)

        self.dropout = torch.nn.Dropout(p=dropout)

    def create_net(self, input_dim, hidden_dim, num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns

    def forward(self, seq, adj, sparse, LP=False):
        graph_output = torch.squeeze(seq, dim=0)
        graph_len = adj
        xs = []
        for i in range(self.num_layers_num):
            input = (graph_output, adj)
            graph_output = self.convs[i](input, sparse)
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = gcn(seq1, adj, sparse)
        # print("h_1",h_1.shape)
        h_3 = h_1 * self.prompt
        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = gcn(seq2, adj, sparse)
        h_4 = h_2 * self.prompt
        ret = self.disc(c, h_3, h_4, samp_bias1, samp_bias2)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)


class DGIprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGIprompt, self).__init__()
        # self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)
        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        seq1 = seq1 * self.prompt
        h_1 = gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        seq2 = seq2 * self.prompt
        h_2 = gcn(seq2, adj, sparse)
        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)


class GraphCL(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCL, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        h_0 = gcn(seq1, adj, sparse)

        h_00 = h_0 * self.prompt
        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        h_11 = h_1 * self.prompt
        h_33 = h_3 * self.prompt

        c_1 = self.read(h_11, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_33, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)

        h_22 = h_2 * self.prompt

        ret1 = self.disc(c_1, h_00, h_22, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_00, h_22, samp_bias1, samp_bias2)

        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)


class GraphCLprompt(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(GraphCLprompt, self).__init__()
        #  self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)

        self.reset_parameters()

    def forward(self, gcn, seq1, seq2, seq3, seq4, adj, aug_adj1, aug_adj2, sparse, msk, samp_bias1, samp_bias2,
                aug_type):

        seq1 = seq1 * self.prompt
        seq2 = seq2 * self.prompt
        seq3 = seq3 * self.prompt
        seq4 = seq4 * self.prompt

        h_0 = gcn(seq1, adj, sparse)

        if aug_type == 'edge':

            h_1 = gcn(seq1, aug_adj1, sparse)
            h_3 = gcn(seq1, aug_adj2, sparse)

        elif aug_type == 'mask':

            h_1 = gcn(seq3, adj, sparse)
            h_3 = gcn(seq4, adj, sparse)

        elif aug_type == 'node' or aug_type == 'subgraph':

            h_1 = gcn(seq3, aug_adj1, sparse)
            h_3 = gcn(seq4, aug_adj2, sparse)

        else:
            assert False

        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        c_3 = self.read(h_3, msk)
        c_3 = self.sigm(c_3)

        h_2 = gcn(seq2, adj, sparse)

        ret1 = self.disc(c_1, h_0, h_2, samp_bias1, samp_bias2)
        ret2 = self.disc(c_3, h_0, h_2, samp_bias1, samp_bias2)
        ret = ret1 + ret2
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)


class LogReg(nn.Module):

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.act = torch.nn.LeakyReLU()
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)
        self.reset_parameters()

    def forward(self, gcn, seq, adj, sparse):
        h_1 = gcn(seq, adj, sparse, True)
        ret = h_1 * self.prompt
        ret = self.sigm(ret.squeeze(dim=0))
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)


class Lpprompt(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lpprompt, self).__init__()
        self.sigm = nn.ELU()
        self.act = torch.nn.LeakyReLU()
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_in), requires_grad=True)
        self.reset_parameters()

    def forward(self, gcn, seq, adj, sparse):
        seq = seq * self.prompt
        h_1 = gcn(seq, adj, sparse, True)
        ret = h_1
        ret = self.sigm(ret.squeeze(dim=0))
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
