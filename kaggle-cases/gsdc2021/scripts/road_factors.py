"""Training-date-excluded line-normal map factors in the trajectory frame."""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from metric import read_positions
from gnss_velocity import ecef

class RoadFactors:
    def __init__(self,root,collection,split,R,origin,excluded_dates=()):
        d=read_positions(root/'artifacts/ground_truth_train.csv')
        exclude=list(excluded_dates) if split=='train' else []
        if split=='train':exclude.append(collection[:10])
        d=d[d.collectionName.str.contains('-SJC-')&~d.collectionName.str[:10].isin(exclude)]
        starts=[];ends=[]
        for c,g in d.groupby('collectionName'):
            g=g[g.phoneName.eq(sorted(g.phoneName.unique())[0])].sort_values('millisSinceGpsEpoch')
            xyz=ecef(np.column_stack([g.latDeg,g.lngDeg,np.zeros(len(g))]))
            points=((xyz-origin)@R.T)[:,:2];a,b=points[:-1],points[1:]
            length=np.linalg.norm(b-a,axis=1);keep=(length>.05)&(length<50)
            starts.append(a[keep]);ends.append(b[keep])
        self.a=np.vstack(starts);self.v=np.vstack(ends)-self.a
        length=np.linalg.norm(self.v,axis=1);step=np.maximum(1,np.ceil(length).astype(int))
        mids=[];ids=[]
        for i in range(len(self.a)):
            fractions=(np.arange(step[i])+.5)/step[i]
            mids.append(self.a[i]+fractions[:,None]*self.v[i]);ids.extend([i]*step[i])
        self.tree=cKDTree(np.vstack(mids));self.ids=np.asarray(ids)

    def factors(self,xy,sigma=1.,limit=40.,huber=5.,direction='axis',heading_fill=False):
        heading=np.gradient(xy,axis=0);speed=np.linalg.norm(heading,axis=1)
        if heading_fill:
            heading=np.gradient(gaussian_filter1d(xy,3,axis=0),axis=0)
            speed=np.linalg.norm(heading,axis=1)
            moving=np.flatnonzero(speed>1.)
            if len(moving):
                index=np.searchsorted(moving,np.arange(len(xy)))
                before=moving[np.clip(index-1,0,len(moving)-1)]
                after=moving[np.clip(index,0,len(moving)-1)]
                nearest=np.where(np.abs(before-np.arange(len(xy)))<np.abs(after-np.arange(len(xy))),before,after)
                heading=heading[nearest];speed=np.linalg.norm(heading,axis=1)
        _,neighbor=self.tree.query(xy,k=min(64 if direction=='signed' else 32,len(self.ids)))
        if neighbor.ndim==1:neighbor=neighbor[:,None]
        si=self.ids[neighbor];a=self.a[si];v=self.v[si]
        f=np.clip(np.sum((xy[:,None,:]-a)*v,axis=2)/np.sum(v*v,axis=2),0,1)
        target=a+f[:,:,None]*v
        dist=np.linalg.norm(target-xy[:,None,:],axis=2)
        align=np.sum(heading[:,None,:]*v,axis=2)/np.maximum(speed[:,None]*np.linalg.norm(v,axis=2),1e-8)
        if direction=='axis':align=np.abs(align)
        cost=dist+100*((align<.65)&(speed[:,None]>1))
        chosen=cost.argmin(axis=1);ix=np.arange(len(xy));direc=v[ix,chosen]
        direc/=np.linalg.norm(direc,axis=1)[:,None]
        normal=np.column_stack([-direc[:,1],direc[:,0]])
        ok=(dist[ix,chosen]<limit)&(cost[ix,chosen]<100)
        weight=ok/sigma*np.sqrt(np.minimum(1,huber/np.maximum(dist[ix,chosen],1e-6)))
        n=len(xy);rows=np.repeat(np.arange(n),2);cols=np.column_stack([np.arange(n),np.arange(n)+n]).ravel()
        C=sparse.csr_matrix((normal.ravel(),(rows,cols)),shape=(n,2*n)).multiply(weight[:,None])
        y=np.sum(normal*target[ix,chosen],axis=1)*weight
        return C,y,int(ok.sum())
