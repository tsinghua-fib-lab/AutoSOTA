from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader




class SFDA2:

    def __init__(self, classifier, config):
        self.classifier = classifier
        self.config = config
        self.batch = config['batch']
        self.workers = config['workers']
        self.bottleneck = config['bottleneck']
        self.num_classes = config['num_classes']

        self.max_iter = config['iter_per_epoch'] * config['epochs']
        self.iter_num = 0
        self.lambda_0 = 5.0
        self.alpha_1 = 1e-4
        self.alpha_2 = 1.0
        self.K = 2

        self.rho = torch.ones([self.num_classes]).cuda() / self.num_classes
        self.cov = torch.zeros(self.num_classes, self.bottleneck, self.bottleneck).cuda()
        self.ave = torch.zeros(self.num_classes, self.bottleneck).cuda()
        self.amount = torch.zeros(self.num_classes).cuda()


    def label_dataset(self, target_train_dl):

        dataset = []
        for i, (images, labels) in tqdm(enumerate(target_train_dl), desc="Creating Indices"):
            indices = torch.arange(i * self.batch, (i + 1) * self.batch)
            for b in range(images.size(0)):
                dataset.append((images[b], indices[b]))
        
        index_dl = DataLoader(dataset, self.batch, shuffle=True, num_workers=self.workers)

        return index_dl


    def create_banks(self, index_dl):

        num_sample = len(index_dl.dataset)
        self.banks = {
            'feature': torch.randn(num_sample, self.bottleneck),
            'output': torch.randn(num_sample, self.num_classes),
            'pseudo': torch.randn(num_sample).long()
        }

        with torch.no_grad():
            self.classifier.train()
            for i, (images, indices) in tqdm(enumerate(index_dl), desc="Creating Banks"):
                images = images.cuda()
                logits, features = self.classifier(images)
                norm_features = F.normalize(features, dim=1)
                outputs = F.softmax(logits, dim=1)
                pseudo_labels = torch.argmax(outputs, dim=1)

                self.banks['feature'][indices] = norm_features.detach().clone().cpu()
                self.banks['output'][indices] = outputs.detach().clone().cpu()
                self.banks['pseudo'][indices] = pseudo_labels.detach().clone().cpu()
        

    def loss(self, images, indices):
        
        logits, features = self.classifier(images)
        norm_features = F.normalize(features, dim=1)
        outputs = F.softmax(logits, dim=1)
        pseudo_labels = torch.argmax(outputs, dim=1)
        alpha = (1 + 10 * self.iter_num / self.max_iter) ** -5

        with torch.no_grad():
            distance = self.banks['feature'][indices] @ self.banks['feature'].T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=self.K+1)
            idx_near = idx_near[:, 1:]
            outputs_near = self.banks['output'][idx_near]

        ## SNC

        rho_batch = torch.histc(pseudo_labels, bins=self.num_classes, min=0, max=self.num_classes - 1) / images.shape[0]
        self.rho = 0.95 * self.rho + 0.05 * rho_batch

        softmax_out_un = outputs.unsqueeze(1).expand(-1, self.K, -1).cuda()

        loss_pos = torch.mean(
            (F.kl_div(softmax_out_un, outputs_near.cuda(), reduction="none").sum(dim=-1)).sum(dim=1)
        )

        mask = torch.ones((images.shape[0], images.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = outputs.T
        dot_neg = outputs @ copy
        dot_neg = ((dot_neg**2) * mask.cuda()).sum(dim=-1)
        neg_pred = torch.mean(dot_neg)
        loss_neg = neg_pred * alpha


        ## IFA

        w = self.classifier.head.weight
        ratio = self.lambda_0 * (self.iter_num / self.max_iter)
        self.update_CV(features, pseudo_labels)
        loss_ifa_ = self.IFA(w, features, logits, ratio)
        loss_ifa = self.alpha_1 * torch.mean(loss_ifa_)


        ## FD

        mean_score = torch.stack([torch.mean(self.banks['output'][self.banks['pseudo'] == i], dim=0) for i in range(self.num_classes)])
        mean_score[mean_score != mean_score] = 0.
        cov_weight = (mean_score @ mean_score.T) * (1.-torch.eye(self.num_classes))
        cov1 = self.cov.view(self.num_classes,-1).unsqueeze(1)
        cov0 = self.cov.view(self.num_classes,-1).unsqueeze(0)
        cov_distance = 1 - torch.sum((cov1*cov0),dim=2) / (torch.norm(cov1, dim=2) * torch.norm(cov0, dim=2) + 1e-12)
        loss_fd = -torch.sum(cov_distance * cov_weight.cuda().detach()) / 2

        self.iter_num += 1

        return loss_pos + loss_neg + (self.alpha_1 * loss_ifa) + (self.alpha_2 * loss_fd)


    def update_CV(self, features, labels):
        N = features.size(0)
        A = features.size(1)
        C = self.num_classes

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A) # mask

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot) # masking

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.ave - ave_CxA).view(C, A, 1),
                (self.ave - ave_CxA).view(C, 1, A)
            )
        )

        self.cov = (self.cov.mul(1 - weight_CV).detach() + var_temp.mul(weight_CV)) + additional_CV
        self.ave = (self.ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.amount = self.amount + onehot.sum(0)


    def IFA(self, w, features, logit, ratio):
        N = features.size(0)
        A = features.size(1)
        C = self.num_classes

        log_prob_ifa_ = []
        sigma2_ = []
        pseudo_labels = torch.argmax(logit, dim=1).detach()
        for i in range(C):
            labels = (torch.ones(N)*i).cuda().long()
            NxW_ij = w.expand(N, C, A)
            NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
            CV_temp = self.cov[pseudo_labels]

            sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij-NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
            with torch.no_grad():
                sigma2_.append(torch.mean(sigma2))
            sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)
            ifa_logit = logit + 0.5 * sigma2
            log_prob_ifa_.append(F.cross_entropy(ifa_logit, labels, reduction='none'))
        log_prob_ifa = torch.stack(log_prob_ifa_)
        loss = torch.sum(2 * log_prob_ifa.T, dim=1)

        return loss