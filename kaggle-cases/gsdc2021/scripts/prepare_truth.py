"""Build training truth only from the official archive, with phone/time keys."""
from pathlib import Path
from io import BytesIO
import zipfile
import pandas as pd
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'data/raw/google-smartphone-decimeter-challenge.zip') as z:
    names=[n for n in z.namelist() if n.startswith('train/') and n.endswith('/ground_truth.csv')]
    assert len(names)==73, len(names)
    frames=[pd.read_csv(BytesIO(z.read(n))) for n in names]
d=pd.concat(frames,ignore_index=True)
d['phone']=d.collectionName+'_'+d.phoneName
assert not d.duplicated(['phone','millisSinceGpsEpoch']).any()
d.to_csv(root/'artifacts/ground_truth_train.csv',index=False)
print('Official truth:',len(d),'rows,',d.phone.nunique(),'phones',flush=True)
