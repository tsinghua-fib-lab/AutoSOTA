import os, pickle, numpy as np
from scipy.stats import linregress
from collections import defaultdict
import torch, sys

M_STREAM=1000; TOKEN_LEN=400; ALPHA=0.05; W0=0.01; GAMMA_EXP=1.2
N_CALIB=20000; PI=0.05; RHO=0.7; TEMP=0.5
TEST_NAMES=["Kolmogorov","Kuiper","Anderson","Cramer","Watson","Chi_squared","Rao","Greenwood"]
DISPLAY={"Kolmogorov":"Kol","Kuiper":"Kui","Anderson":"And","Cramer":"Cra","Watson":"Wat","Chi_squared":"Chi","Rao":"Ney","Greenwood":"Phi"}

class OnlineLORD:
    def __init__(self):
        self.alpha=ALPHA; self.w0=W0
        raw=np.arange(1,M_STREAM+5001)**-GAMMA_EXP
        self.gamma=0.07*raw; self.gamma/=self.gamma.sum()
        self.last_disc=0; self.wealth=W0
    def test(self,t,p):
        d=(t+1)-self.last_disc
        at=self.wealth*self.gamma[d] if d<len(self.gamma) else 1e-20
        r=p<=at
        if r: self.wealth+=(self.alpha-self.w0); self.last_disc=t+1
        else: self.wealth-=at
        if self.wealth<0: self.wealth=0
        return r

class NaiveFixed:
    def __init__(self): self.alpha=ALPHA
    def test(self,t,p): return p<=self.alpha

class GoF:
    @staticmethod
    def kolmogorov(Y):
        n=len(Y); Y=np.sort(Y); r=np.arange(1,n+1)
        return max(np.maximum(r/n-Y,Y-(r-1)/n).max(),0)
    @staticmethod
    def kuiper(Y):
        n=len(Y); Y=np.sort(Y); r=np.arange(1,n+1)
        return np.max(r/n-Y)+np.max(Y-(r-1)/n)
    @staticmethod
    def anderson(Y):
        n=len(Y); Y=np.sort(Y); Y=np.clip(Y,1e-10,1-1e-10)
        S=np.sum((2*np.arange(1,n+1)-1)*(np.log(Y)+np.log(1-Y[::-1])))/n
        return -n-S
    @staticmethod
    def cramer(Y):
        n=len(Y); Y=np.sort(Y)
        return 1/(12*n)+np.sum((Y-(2*np.arange(1,n+1)-1)/(2*n))**2)
    @staticmethod
    def watson(Y):
        return GoF.cramer(Y)-len(Y)*(np.mean(Y)-0.5)**2
    @staticmethod
    def chi_squared(Y,c=10):
        obs,_=np.histogram(Y,bins=np.linspace(0,1,c+1))
        return np.sum((obs-len(Y)/c)**2/(len(Y)/c))
    @staticmethod
    def rao(Y):
        n=len(Y); Y=np.sort(Y); sp=np.diff(Y,prepend=0); sp=np.append(sp,1-Y[-1])
        return 0.5*n*np.sum(np.abs(sp-1/(n+1)))
    @staticmethod
    def greenwood(Y):
        n=len(Y); Y=np.sort(Y); sp=np.diff(Y,prepend=0); sp=np.append(sp,1-Y[-1])
        return np.sum(sp**2)

def calc_scores(Y):
    return {
        "Kolmogorov":GoF.kolmogorov(Y),"Kuiper":GoF.kuiper(Y),
        "Anderson":GoF.anderson(Y),"Cramer":GoF.cramer(Y),
        "Watson":GoF.watson(Y),"Chi_squared":GoF.chi_squared(Y),
        "Rao":GoF.rao(Y),"Greenwood":GoF.greenwood(Y),
    }

class Calibrator:
    def __init__(self,scores):
        self.scores=np.sort(scores); self.n=len(scores)
        t=int(self.n*0.9); self.tail=self.scores[t:]; self.thresh=self.scores[t]
        ep=(self.n-np.arange(t+1,self.n+1)+1)/(self.n+1)
        r=linregress(self.tail,np.log(ep)); self.slope=r.slope; self.intercept=r.intercept
    def pval(self,s):
        if s<=self.thresh:
            i=np.searchsorted(self.scores,s,side="left"); return (self.n-i+1)/(self.n+1)
        return np.exp(self.slope*s+self.intercept)

def build_calibs():
    np.random.seed(42)
    store=defaultdict(list)
    for _ in range(N_CALIB):
        y=np.random.rand(TOKEN_LEN)
        scores=calc_scores(y)
        for k,v in scores.items(): store[k].append(v)
    return {k:Calibrator(v) for k,v in store.items()}

path="raw_data/opt1.3b_temp_0.5_len_400_cnt_1000.pkl"
d=pickle.load(open(path,"rb"))
Y=d["watermark"]["Ys"]
if torch.is_tensor(Y): Y=Y.cpu().numpy()
pool=Y

calibs=build_calibs()
sys.stdout.flush()

paper_online = {
    "Kol":(0.021,0.885),"Kui":(0.000,0.846),"And":(0.038,0.981),
    "Cra":(0.059,0.923),"Wat":(0.048,0.769),"Chi":(0.000,0.885),
    "Ney":(0.029,0.635),"Phi":(0.000,0.212),
}

for SEED in [42, 999, 2025]:
    np.random.seed(SEED)
    labels=np.random.choice([0,1],size=M_STREAM,p=[1-PI,PI])
    n_real=int(np.sum(labels))
    
    methods=[f"{p}_{tn}" for tn in TEST_NAMES for p in ["Naive","Lord"]]
    ctrls={m: NaiveFixed() if m.startswith("Naive") else OnlineLORD() for m in methods}
    stats={m:{"tp":0,"fp":0} for m in methods}
    
    for t in range(M_STREAM):
        if labels[t]==1:
            Yp=pool[np.random.randint(len(pool))]
            mask=np.random.rand(TOKEN_LEN)<RHO
            Yd=np.random.rand(TOKEN_LEN); Yd[mask]=Yp[mask]; Yf=Yd
        else:
            Yf=np.random.rand(TOKEN_LEN)
        scores=calc_scores(Yf)
        for tn in TEST_NAMES:
            pval=calibs[tn].pval(scores[tn])
            for mode in ["Naive","Lord"]:
                k=f"{mode}_{tn}"
                if ctrls[k].test(t,pval):
                    if labels[t]==1: stats[k]["tp"]+=1
                    else: stats[k]["fp"]+=1
    
    print(f"\n=== SEED={SEED} n_real={n_real} ===")
    print('{:6s} | {:>9s} {:>9s} | {:>9s} {:>9s} | {:>11s} {:>11s}'.format('Stat','Naive-FDR','Naive-Pow','LORD-FDR','LORD-Pow','Paper-L-FDR','Paper-L-Pow'))
    print("-"*78)
    for tn in TEST_NAMES:
        d=DISPLAY[tn]
        nf = stats[f"Naive_{tn}"]["fp"]/(stats[f"Naive_{tn}"]["tp"]+stats[f"Naive_{tn}"]["fp"]) if (stats[f"Naive_{tn}"]["tp"]+stats[f"Naive_{tn}"]["fp"])>0 else 0
        npow = stats[f"Naive_{tn}"]["tp"]/n_real if n_real>0 else 0
        lf = stats[f"Lord_{tn}"]["fp"]/(stats[f"Lord_{tn}"]["tp"]+stats[f"Lord_{tn}"]["fp"]) if (stats[f"Lord_{tn}"]["tp"]+stats[f"Lord_{tn}"]["fp"])>0 else 0
        lpow = stats[f"Lord_{tn}"]["tp"]/n_real if n_real>0 else 0
        pf, pp = paper_online[d]
        close = " <<<" if abs(lf-pf)<0.03 and abs(lpow-pp)<0.06 else ""
        print("{:<6} | {:9.4f} {:9.4f} | {:9.4f} {:9.4f} | {:11.3f} {:11.3f}{}".format(d,nf,npow,lf,lpow,pf,pp,close))
    sys.stdout.flush()
