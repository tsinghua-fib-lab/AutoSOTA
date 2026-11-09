"""Automatic SJC map matching against other training drives, with date exclusion."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import spsolve
from gnss_velocity import ecef,enu_matrix

def match(pred,truth,alpha=1.0,limit=10.0,training=False,excluded_dates=(),mode='snap',drift_sigma=.05,road_sigma=.5,anchor_sigma=20.):
    output=pred.copy();R=enu_matrix(37.335,-121.89);origin=ecef(np.array([[37.335,-121.89,0.]]))[0]
    def xy(d):return ((ecef(np.column_stack([d.latDeg,d.lngDeg,np.zeros(len(d))]))-origin)@R.T)[:,:2]
    for collection,drive in pred.groupby('collectionName'):
        if '-SJC-' not in collection:continue
        refs=truth[truth.collectionName.str.contains('-SJC-')].copy()
        dates=list(excluded_dates)
        if training:dates.append(collection[:10])
        refs=refs[~refs.collectionName.str[:10].isin(dates)]
        starts=[];ends=[]
        for c,g in refs.groupby('collectionName'):
            # One GNSS/INS ground-truth trajectory per reference vehicle.
            g=g[g.phoneName.eq(sorted(g.phoneName.unique())[0])].sort_values('millisSinceGpsEpoch')
            points=xy(g);a,b=points[:-1],points[1:];length=np.linalg.norm(b-a,axis=1)
            valid=(length>.05)&(length<50)
            a,b=a[valid],b[valid]
            if len(a):starts.append(a);ends.append(b)
        if not starts:continue
        a=np.vstack(starts);b=np.vstack(ends);v=b-a
        # Subdivide segments before nearest-neighbor lookup to avoid missing
        # long segments; each lookup returns the actual segment identity.
        lengths=np.linalg.norm(v,axis=1);step=np.maximum(1,np.ceil(lengths/1.).astype(int))
        mids=[];ids=[]
        for i in range(len(a)):
            frac=(np.arange(step[i])+.5)/step[i]
            mids.append(a[i]+frac[:,None]*v[i]);ids.extend([i]*step[i])
        mids=np.vstack(mids);ids=np.asarray(ids);tree=cKDTree(mids)
        for phone,g in drive.groupby('phone'):
            g=g.sort_values('millisSinceGpsEpoch');q=xy(g)
            heading=np.gradient(q,axis=0);speed=np.linalg.norm(heading,axis=1)
            correction=np.zeros_like(q)
            for iteration in range(8 if mode=='road' else 1):
                current=q+correction
                _,neighbors=tree.query(current,k=min(32,len(mids)))
                if neighbors.ndim==1:neighbors=neighbors[:,None]
                si=ids[neighbors];va=v[si];aa=a[si]
                fraction=np.clip(np.sum((current[:,None,:]-aa)*va,axis=2)/np.sum(va**2,axis=2),0,1)
                projected=aa+fraction[:,:,None]*va
                distance=np.linalg.norm(projected-current[:,None,:],axis=2)
                alignment=np.abs(np.sum(heading[:,None,:]*va,axis=2))/np.maximum(speed[:,None]*np.linalg.norm(va,axis=2),1e-8)
                cost=distance+((alignment<.65)&(speed[:,None]>1.))*100
                idx=cost.argmin(axis=1);chosen=projected[np.arange(len(q)),idx];dist=distance[np.arange(len(q)),idx]
                valid=(dist<limit)&(cost[np.arange(len(q)),idx]<100)
                if mode=='snap':
                    correction=(chosen-q)*alpha;correction[~valid]=0
                elif mode=='road':
                    direction=va[np.arange(len(q)),idx];direction/=np.linalg.norm(direction,axis=1)[:,None]
                    normal=np.column_stack([-direction[:,1],direction[:,0]])
                    count=len(q);rows=np.repeat(np.arange(count),2);cols=np.column_stack([np.arange(count),np.arange(count)+count]).ravel()
                    weight=valid.astype(float)/road_sigma*np.sqrt(np.minimum(1.,5./np.maximum(dist,1e-6)))
                    C=sparse.csr_matrix((normal.ravel(),(rows,cols)),shape=(count,2*count)).multiply(weight[:,None])
                    target=np.sum(normal*(chosen-q),axis=1)*weight
                    dt=np.maximum(np.diff(g.millisSinceGpsEpoch.to_numpy())/1000,.01)
                    D=sparse.diags([-np.ones(count-1),np.ones(count-1)],[0,1],shape=(count-1,count),format='csr').multiply((1/(drift_sigma*np.sqrt(dt)))[:,None])
                    smooth=sparse.block_diag([D,D],format='csr')
                    system=C.T@C+smooth.T@smooth+sparse.eye(2*count)/anchor_sigma**2
                    fitted=spsolve(system.tocsc(),C.T@target)
                    correction=np.column_stack([fitted[:count],fitted[count:]])
                    correction*=np.minimum(1,60/np.maximum(np.linalg.norm(correction,axis=1),1e-9))[:,None]
                else:raise ValueError(mode)
            latrad=np.radians(g.latDeg.to_numpy());e2=6.6943799901413165e-3;major=6378137.
            M=major*(1-e2)/(1-e2*np.sin(latrad)**2)**1.5;N=major/np.sqrt(1-e2*np.sin(latrad)**2)
            output.loc[g.index,'latDeg']=g.latDeg.to_numpy()+np.degrees(correction[:,1]/M)
            output.loc[g.index,'lngDeg']=g.lngDeg.to_numpy()+np.degrees(correction[:,0]/(N*np.cos(latrad)))
    return output
