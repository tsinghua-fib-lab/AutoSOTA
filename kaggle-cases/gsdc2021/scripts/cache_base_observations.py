"""Cache public SLAC code observations from the author's RINEX release."""
from pathlib import Path
import argparse
import json
import numpy as np
import georinex as gr
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
RELEASE=ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google'

def cache(path):
    out=ROOT/'artifacts/base_observations'/path.parent.parent.name/path.parent.name/f'{path.name}.npz';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return {'file':path.name,'status':'cached'}
    d=gr.load(path,meas=['C1','C5'],use={'G','R','E'})
    arrays={'time_ms':((d.time.values.astype('datetime64[ms]')-np.datetime64('1980-01-06','ms'))/np.timedelta64(1,'ms')).astype('int64'),'sv':d.sv.values.astype('U5'),'position':np.asarray(d.attrs['position'],float)}
    for name in ['C1','C5']:
        if name in d:arrays[name]=d[name].values
    np.savez_compressed(out,**arrays)
    return {'file':path.name,'status':'ok','epochs':len(d.time),'satellites':len(d.sv)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args()
    paths=[p for split in ['train','test'] for p in (RELEASE/split).glob('*/slac*.*o')]
    if a.limit:paths=paths[:a.limit]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(cache,p) for p in paths]):
            r=future.result();results.append(r);print(json.dumps(r),flush=True)
    (ROOT/'artifacts/base_observation_status.json').write_text(json.dumps(results,indent=2))
