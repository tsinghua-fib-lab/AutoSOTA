"""Repair the released PPK CSV headers without changing any predictions."""
from pathlib import Path
import json
import pandas as pd
from metric import KEY
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google'
cols=['collectionName','phoneName','millisSinceGpsEpoch','latDeg','lngDeg','heightAboveWgs84EllipsoidM','stde','stdn','stdu','Q','ns','phone']
info=[]
for split in ['train','test']:
    src=base/f'baseline_locations_{split}_12_30.csv'
    # Author's header lists 7 fields, but each observation contains 12 fields.
    d=pd.read_csv(src,skiprows=1,header=None,names=cols,skipinitialspace=True)
    d['phone']=d.phone.str.strip()
    if not (d.phone==d.collectionName+'_'+d.phoneName).all():raise ValueError('Phone identity mismatch')
    if d.duplicated(KEY).any():raise ValueError('Duplicate reference keys')
    official=pd.read_csv(ROOT/f'data/competition/baseline_locations_{split}.csv')
    audit=official[KEY].merge(d[KEY],on=KEY,how='outer',indicator=True,validate='one_to_one')
    if not audit._merge.eq('both').all():raise ValueError('Reference key mismatch')
    dst=ROOT/f'artifacts/rtklib_author_{split}.csv';d.to_csv(dst,index=False)
    info.append(dict(source=str(src.relative_to(ROOT)),rows=len(d),header_repaired=True))
(ROOT/'artifacts/rtklib_header_repairs.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
