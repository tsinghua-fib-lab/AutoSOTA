import math
import torch
import torch.nn as nn
from torch_basic_settings import DEVICE
from optimizers.utils import sortIndivBND
from optimizers.base_model import GradTrainedBBO


class SMBND(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batchpop1, batchpop2, minimize=True):
        """Selection operator used by POM.

        Args:
            batchpop1: offspring population, shape (B, N, 1 + D)
            batchpop2: parent population, shape (B, N, 1 + D)
            minimize: if True, lower fitness is better
        """

        b, n, d = batchpop1.shape
        fit1 = batchpop1[..., 0]  #
        fit2 = batchpop2[..., 0]
        batchMask = fit1 - fit2  # b,n
        if minimize:
            batchMask[batchMask >= 0] = 0
            batchMask[batchMask < 0] = 1

        else:
            batchMask[batchMask <= 0] = 0
            batchMask[batchMask > 0] = 1

        batchMask = torch.unsqueeze(batchMask, -1)  # b,n,1
        batchMask = batchMask.repeat(1, 1, d)
        batchMask1 = torch.ones_like(batchMask).to(DEVICE) - batchMask
        nextPop = batchpop1 * batchMask + batchpop2 * batchMask1

        return nextPop


class MSA(nn.Module):
    """
    Scaled dot-product attention (fixed parameterization).
    """

    def __init__(self, dim=3, dh=10, dq=8, dk=8, dv=8):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(dim, dh), nn.LeakyReLU(), nn.LayerNorm(dh))
        self.fq = nn.Sequential(
            nn.Linear(dh, dq)
        )  # ,nn.LeakyReLU(),nn.LayerNorm(dq),nn.Linear(dq,dq))
        self.fk = nn.Sequential(
            nn.Linear(dh, dk)
        )  # ,nn.LeakyReLU(),nn.LayerNorm(dk),nn.Linear(dk,dk))
        self.fv = nn.Sequential(
            nn.Linear(dh, dv)
        )  # ,nn.LeakyReLU(),nn.LayerNorm(dv),nn.Linear(dv,dv))

    def forward(self, x):
        b, n, d = x.shape
        x = self.w(x)
        q = self.fq(x)
        k = self.fk(x)
        v = self.fv(x)
        a = torch.matmul(q, torch.transpose(k, -1, -2)) / torch.sqrt(
            torch.tensor(k.shape[-1])
        )
        a = torch.softmax(a, dim=-1)
        vs = torch.matmul(a, v).view(b, n, -1)
        return vs


class GBMutModel(nn.Module):
    def __init__(self, hdim=1000):
        super().__init__()
        fdim = 2
        hdim2 = 100
        qkdim = hdim
        self.fq1 = nn.Sequential(
            nn.Linear(hdim2, qkdim),
            nn.Tanh(),
        )

        self.fk1 = nn.Sequential(
            nn.Linear(hdim2, qkdim),
            nn.Tanh(),
        )

        self.w = nn.Sequential(
            nn.Linear(fdim, hdim2),
            nn.Tanh(),
            # nn.LayerNorm(hdim2)
        )

    def forward(self, x):

        # b,n,2
        x = self.w(x)
        q1 = self.fq1(x)
        k1 = self.fk1(x)
        A = torch.matmul(q1, torch.transpose(k1, -1, -2)) / torch.sqrt(
            torch.tensor(k1.shape[-1]).to(DEVICE)
        )
        A = torch.tanh(A)
        mask = torch.rand_like(A).to(DEVICE)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        y = torch.eye(mask.shape[-2], mask.shape[-1]).to(DEVICE)
        y = torch.unsqueeze(y, 0)
        mask = mask + y
        mask[mask == 2] = 1
        A = A * mask
        return A


class GBLearnCrRate(nn.Module):
    def __init__(self, hdim=100):
        super().__init__()
        inputdim = 3
        outputdim = 1
        self.net = nn.Sequential(
            nn.Linear(inputdim, hdim),
            nn.ReLU(),
            nn.LayerNorm(hdim),
            nn.Linear(hdim, outputdim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.net(x)  # b,n,dim
        return x


class GeneBasedCrossover(nn.Module):
    def __init__(self, dim=2, dh=10, dq=10, dk=10, dv=10):
        super().__init__()
        self.msa = MSA(dim=dim, dh=dh, dq=dq, dk=dk, dv=dv)
        self.ffn = nn.Sequential(
            nn.Linear(dv, dv), nn.ReLU(), nn.LayerNorm(dv), nn.Linear(dv, 2)
        )

    def forward(self, x, fittoken):
        """
        Args:
            x: (B, N, 2 * D) concatenated parent and offspring chromosomes.
            fittoken: (B, N, 2) parent fitness and rank tokens.
        """
        b, n, d = x.shape
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        x = (x - mean) / std
        x = x.view(b * n, 2, -1)
        x = torch.permute(x, [0, 2, 1])
        fittoken = fittoken.view((b * n, 1, 2))
        x = torch.cat((x, fittoken), dim=-2)
        x = self.msa(x)  # b*n,-1,dv
        x = self.ffn(x)  # b*n,-1,2
        x = x.view(b, n, -1, 2)
        x = x[..., :-1, :]
        # x=x-torch.min(x,dim=-1,keepdim=True)[0]
        x = torch.softmax(x, dim=-1)  # b,n,d,2
        return x


# Pre-trained Optimization Model (POM).
class POM(GradTrainedBBO):
    def __init__(
        self,
        config=None,
        popsize=100,
        selmod="1-to-1",
        cr_policy="learned",
        muthdim=1000,
        crhdim=4,
    ):
        super().__init__()
        self.name = "POM"

        # Config-driven init (for MetaDistill trainers / eval scripts).
        if isinstance(config, dict):
            cfg = config
            popsize = int(cfg.get("popsize", popsize))
            selmod = cfg.get("selmod", selmod)
            cr_policy = cfg.get("cr_policy", cr_policy)
            muthdim = int(cfg.get("muthdim", muthdim))
            crhdim = int(cfg.get("crhdim", crhdim))
            self.seed = cfg.get("seed", None)
            # Mutation scaling / stabilization options (default: disabled for backward compatibility).
            self.mut_scale_mode = str(cfg.get("mut_scale_mode", "none")).lower()
            self.mut_scale_eps = float(cfg.get("mut_scale_eps", 1e-8))
            # If enabled, return additional non-detached tensors for trainer-side penalties/diagnostics.
            self.return_aux = bool(cfg.get("return_aux", False))
        else:
            # Backward compatibility: allow positional popsize as the first arg.
            if config is not None:
                popsize = config
            self.seed = None
            self.mut_scale_mode = "none"
            self.mut_scale_eps = 1e-8
            self.return_aux = False

        popsize = int(popsize)
        self.popsize = popsize
        self.cr_policy = cr_policy
        self.ranks = (
            (torch.arange(0, popsize, requires_grad=False).to(DEVICE))
            .float()
            .to(DEVICE)
        )
        self.ranks = (
            self.ranks - torch.mean(self.ranks, dim=-1, keepdim=True)
        ) / torch.std(self.ranks, dim=-1, keepdim=True)
        self.ranks = self.ranks.view(1, -1, 1)
        self.ranks2 = (
            (torch.arange(0, popsize * 2 - 1, requires_grad=False).to(DEVICE))
            .float()
            .to(DEVICE)
        )
        self.ranks2 = (
            self.ranks2 - torch.mean(self.ranks2, dim=-1, keepdim=True)
        ) / torch.std(self.ranks2, dim=-1, keepdim=True)
        self.ranks2 = self.ranks2.view(1, -1, 1)
        self.sm = SMBND()
        self.agen = GBMutModel(hdim=muthdim)
        self.crgen = GBLearnCrRate(hdim=crhdim)
        sel = str(selmod).strip().lower()
        if sel in {"1to1", "one-to-one", "one2one"}:
            sel = "1-to-1"
        if sel in {"roulette-wheel", "wheel"}:
            sel = "roulette"
        if sel == "\u8f6e\u76d8\u8d4c":
            sel = "roulette"
        self.selmod = sel
        self.adapter = None

    def _apply_mut_scale(self, params: torch.Tensor, *, popsize: int) -> torch.Tensor:
        """Stabilize mutation mixing matrix scale to avoid vchrom explosion for large popsize.

        Supported modes:
        - none: no scaling (original behavior)
        - sqrt_pop: divide by sqrt(popsize)
        - row_l1: normalize each row by its L1 norm (abs-sum)
        - row_l2: normalize each row by its L2 norm
        """
        mode = str(getattr(self, "mut_scale_mode", "none")).lower()
        eps = float(getattr(self, "mut_scale_eps", 1e-8))
        if mode in {"none", "off", "disable", "disabled"}:
            return params

        popsize = int(max(1, popsize))
        if mode in {"sqrt_pop", "sqrt_n", "sqrt"}:
            return params / math.sqrt(popsize)

        if mode in {"row_l1", "l1"}:
            denom = params.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
            return params / denom

        if mode in {"row_l2", "l2"}:
            denom = torch.sqrt((params * params).sum(dim=-1, keepdim=True)).clamp_min(eps)
            return params / denom

        raise ValueError(f"Unknown mut_scale_mode: {mode}")

    def set_seed(self, seed: int):
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def setAdapter(self, adapter):
        self.adapter = adapter

    def RoulleteSelectWithElite(self, pop, popsize):
        """
        Roulette-wheel selection with elite preservation.

        Args:
            pop: population tensor of shape (B, N, 1 + D)
            popsize: number of selected individuals
        """
        b, n, d = pop.shape
        fitness = pop[..., 0]  # b,n
        p = 1 - torch.softmax(fitness, dim=-1)
        selected_index = torch.multinomial(p, popsize, replacement=False)
        offs = []
        for idx, batchpop in enumerate(pop):
            index = selected_index[idx]
            tmp = batchpop[index]
            tmp = torch.unsqueeze(tmp, 0)
            offs.append(tmp)
        offs = torch.cat(offs)
        return offs

    def genGBMutToken(self, x, ranks):
        b, n = x.shape  # b,n
        # x=self.fln(x).view(b,-1,1)
        miu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        x = (x - miu) / std
        x = x.view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks[..., :n, :]), dim=-1)  # (b,w*h,2)
        return x

    def genGBMutToken2(self, x, ranks):
        b, _ = x.shape  # b,n
        x = self.fln2(x).view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks), dim=-1)  # (b,w*h,2)
        return x

    def genCrRankToken(self, fitness):
        """
        input:b,n
        """
        b, n = fitness.shape
        _, indexs = torch.sort(fitness, dim=-1)  # b,n
        ranks = self.ranks.repeat(b, 1, 1)
        # fitness=self.fln(fitness)
        miu = torch.mean(fitness, dim=-1, keepdim=True)
        std = torch.std(fitness, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        fitness = (fitness - miu) / std
        fitness = fitness.view(b, -1, 1)
        newRanks = []
        for bid in range(b):
            index = indexs[bid, ...]
            index = torch.unsqueeze(index, -1)
            rank = ranks[bid, ...]
            tmp = torch.cat((index, rank[:n, ...]), dim=-1)
            _, tmp_index = torch.sort(tmp[:, 0], dim=0)
            tmp = torch.index_select(tmp, 0, tmp_index)
            token = torch.cat((fitness[bid], torch.unsqueeze(tmp[:, 1], -1)), -1)
            newRanks.append(token)
        token = torch.stack(newRanks, 0)
        return token

    def genCrRankTokenWithoutFit(self, father, off):
        """
        Build crossover rank token without using fitness.
        """
        b, n, d = father.shape
        # cosine similarity between father and offspring
        tmp = torch.cat((father, off), dim=-1)  # b,n,2d
        ave = torch.mean(tmp, dim=-1, keepdim=True)  # b,n,1
        std = torch.std(tmp, dim=-1, keepdim=True)  # b,n,1
        tmp = (tmp - ave) / (std + 1e-8)
        fpop = tmp[:, :, :d]
        opop = tmp[:, :, d:]
        fMod = torch.sqrt(torch.sum(fpop**2, dim=-1, keepdim=True))  # b,n,1
        oMod = torch.sqrt(torch.sum(opop**2, dim=-1, keepdim=True))  # b,n,1
        item = fMod * oMod
        item = torch.clamp(item, min=1e-8)
        sim = torch.sum(fpop * opop, dim=-1, keepdim=True) / (item)  # b,n,1
        # sim=torch.sum(father*off,dim=-1,keepdim=True)/(item) #b,n,1
        sim = sim.view(b, n)
        # sim=torch.softmax(sim,dim=-1)
        sim = (sim - torch.mean(sim, dim=-1, keepdim=True)) / torch.std(
            sim, dim=-1, keepdim=True
        )
        sim = torch.unsqueeze(sim, -1)

        return sim

    def clearMutstate(self):
        self.gbmut.resetSigma()
        self.improvedFlag = None

    def forward(self, batchPop=None, problem=None):
        """
        One optimization step.

        Args:
            batchPop: population with fitness, shape (B, N, 1 + D)
            problem: task wrapper providing calfitness()
        """
        paramcr = None
        b, n, d = batchPop.shape
        batchPop = sortIndivBND(batchPop)
        batchfitness = batchPop[:, :, 0]
        fitnesstoken = self.genGBMutToken(batchfitness, self.ranks)  # b,n,2
        Atoken = fitnesstoken
        params_raw = self.agen(Atoken)
        params = self._apply_mut_scale(params_raw, popsize=n)
        batchChrom = batchPop[:, :, 1:]
        # params=torch.zeros((1,n,n),device=DEVICE)
        # for i in range(n):
        #     idx=np.random.choice(n,3,False)
        #     params[0,i,idx[0]]=1
        #     params[0,i,idx[1]]=-0.5
        #     params[0,i,idx[2]]=0.5
        vchrom = torch.matmul(params, batchChrom)
        # Crossover
        if self.cr_policy == "learned":
            # vpop,r1=problem.calfitness(vchrom)
            # r1=torch.squeeze(r1,-1)
            # vpopToken=self.genCrRankToken(r1)
            vpopToken = self.genCrRankTokenWithoutFit(batchChrom, vchrom)
            token = torch.cat((fitnesstoken, vpopToken), dim=-1)  # b,n,4
            cr = self.crgen(token)  # b,n,1
            cr = torch.unsqueeze(cr, -2)  # b,n,1,1
            paramcr = cr.view(
                b,
                n,
            )
            cr = cr.repeat(1, 1, batchChrom.shape[-1], 1)  # b,n,d,1
            r = torch.rand_like(cr).to(DEVICE)
            select_mask = torch.cat((cr, r), dim=-1)  # b,n,d,2
            select_mask = torch.nn.functional.gumbel_softmax(
                select_mask, tau=1, hard=True, eps=1e-10, dim=-1
            )
            offpopChrom = (
                select_mask[..., 0] * batchChrom + select_mask[..., 1] * vchrom
            )

        else:
            mask = torch.rand_like(batchChrom).to(DEVICE)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            offpopChrom = mask * batchChrom + (1 - mask) * vchrom

        if self.adapter:
            offpopChrom = self.adapter(offpopChrom)

        offpopChrom_preclip = offpopChrom
        offpop, _ = problem.calfitness(offpopChrom)  # b,n,d
        # Selection
        mixpop = torch.cat((batchPop, offpop), dim=1)
        if self.selmod == "learned":
            pass
            # mixpop=sortIndivBND(mixpop)
            # elitePop=mixpop[:,:1,:]
            # mixpop=mixpop[:,1:,:]
            # mixPopfitness=mixpop[...,0]
            # mixChrom=mixpop[...,1:]
            # mixToken=self.genGBMutToken2(mixPopfitness,self.ranks2)  #b,n,2
            # batchMod=torch.sqrt(torch.sum(mixChrom**2,dim=-1,keepdim=True))  #b,n,1
            # batchModMatrix=torch.matmul(batchMod,torch.transpose(batchMod,-1,-2)) #b,n,n
            # crossdistance=torch.matmul(mixChrom,torch.transpose(mixChrom,-1,-2))/batchModMatrix #b,n,n
            # crossdistance=torch.softmax(crossdistance,dim=-1)
            # stoken=torch.cat((mixToken,crossdistance),dim=-1)
            # selectednextPop=self.selmodel(stoken,mixpop)
            # nextPop=torch.cat((elitePop,selectednextPop),dim=1)

        if self.selmod == "1-to-1":
            nextPop = self.sm(offpop, batchPop)

        if self.selmod == "roulette":
            mixpop = sortIndivBND(mixpop)
            elitePop = mixpop[:, :1, :]
            mixpop = mixpop[:, 1:, :]
            selectednextPop = self.RoulleteSelectWithElite(
                mixpop, batchPop.shape[1] - 1
            )
            nextPop = torch.cat((elitePop, selectednextPop), dim=1)

        info = {
            "fitness": nextPop[..., 0:1].detach(),
            "s": params.detach(),
            "cr": paramcr.detach(),
        }
        if getattr(self, "return_aux", False):
            info["aux"] = {
                "mut_params_raw": params_raw,
                "mut_params": params,
                "vchrom": vchrom,
                "offpop_chrom": offpopChrom_preclip,
            }

        return nextPop, info


class POMNoGumbel(GradTrainedBBO):
    def __init__(self, popsize=100, selmod="1-to-1", muthdim=1000):
        super().__init__()
        self.name = "POMNoGumbel"
        self.needs_backward = True

        self.popsize = popsize
        self.ranks = (
            (torch.arange(0, popsize, requires_grad=False).to(DEVICE))
            .float()
            .to(DEVICE)
        )
        self.ranks = (
            self.ranks - torch.mean(self.ranks, dim=-1, keepdim=True)
        ) / torch.std(self.ranks, dim=-1, keepdim=True)
        self.ranks = self.ranks.view(1, -1, 1)
        # self.ranks2=(torch.arange(0,popsize*2-1,requires_grad=False).to(DEVICE)).float().to(DEVICE)
        # self.ranks2=(self.ranks2-torch.mean(self.ranks2,dim=-1,keepdim=True))/torch.std(self.ranks2,dim=-1,keepdim=True)
        # self.ranks2=self.ranks2.view(1,-1,1)
        self.sm = SMBND()
        self.agen = GBMutModel(hdim=muthdim)
        # self.crgen=GBLearnCrRate(hdim=crhdim)
        self.lcm = GeneBasedCrossover(dim=2, dh=10, dq=20, dk=20, dv=20)
        sel = str(selmod).strip().lower()
        if sel in {"1to1", "one-to-one", "one2one"}:
            sel = "1-to-1"
        if sel in {"roulette-wheel", "wheel"}:
            sel = "roulette"
        if sel == "\u8f6e\u76d8\u8d4c":
            sel = "roulette"
        self.selmod = sel
        # self.adapter=None

    def setAdapter(self, adapter):
        self.adapter = adapter

    def genGBMutToken(self, x, ranks):
        b, n = x.shape  # b,n
        # x=self.fln(x).view(b,-1,1)
        miu = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        x = (x - miu) / std
        x = x.view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks[..., :n, :]), dim=-1)  # (b,w*h,2)
        return x

    def forward(self, batchPop=None, problem=None):
        """
        One optimization step.

        Args:
            batchPop: population with fitness, shape (B, N, 1 + D)
            problem: task wrapper providing calfitness()
        """
        b, n, d = batchPop.shape
        batchPop = sortIndivBND(batchPop)
        batchfitness = batchPop[:, :, 0]
        fitnesstoken = self.genGBMutToken(batchfitness, self.ranks)  # b,n,2
        Atoken = fitnesstoken
        params = self.agen(Atoken)

        batchChrom = batchPop[:, :, 1:]
        # params=torch.zeros((1,n,n),device=DEVICE)
        # for i in range(n):
        #     idx=np.random.choice(n,3,False)
        #     params[0,i,idx[0]]=1
        #     params[0,i,idx[1]]=-0.5
        #     params[0,i,idx[2]]=0.5
        vchrom = torch.matmul(params, batchChrom)

        # Crossover
        X = torch.cat((batchChrom, vchrom), dim=-1)
        crp = self.lcm(X, fitnesstoken)
        offpopChrom = batchChrom * crp[..., 0] + vchrom * crp[..., 1]
        offpop, _ = problem.calfitness(offpopChrom)  # b,n,d
        # Selection
        mixpop = torch.cat((batchPop, offpop), dim=1)

        if self.selmod == "1-to-1":
            nextPop = self.sm(offpop, batchPop)

        info = {
            "fitness": nextPop[..., 0:1].detach(),
            "s": params.detach(),
            "cr": crp.detach(),
        }

        return nextPop, info


if __name__ == "__main__":
    lcm = GeneBasedCrossover(dim=2, dh=10, dq=10, dk=10, dv=10)
    b, n, d = (2, 10, 5)
    x = torch.rand((b, n, d * 2))
    fittoken = torch.rand((b, n, 2))
    y = lcm(x, fittoken)
    print(y)
