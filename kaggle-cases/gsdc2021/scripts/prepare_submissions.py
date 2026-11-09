"""Canonicalize reference outputs against official sample submission keys."""
from pathlib import Path
import hashlib
import json
from metric import read_positions,validate_submission,KEY,COORD
ROOT=Path(__file__).resolve().parents[1]
sample=read_positions(ROOT/'data/competition/sample_submission.csv')
sources={
 'rtklib_author_reference':ROOT/'upstream/demo5_b34e_phone/demo5_b34e_phone/Google/submission_1230.csv',
 'saito_original_smoothing':ROOT/'artifacts/saito_smoothing/smoothing_1st_test.csv',
}
info=[]
for name,path in sources.items():
    pred=read_positions(path)
    sub=sample[KEY].merge(pred[KEY+COORD],on=KEY,how='left',validate='one_to_one')
    if len(pred)!=len(sample):raise ValueError('Extra or missing prediction keys')
    validate_submission(sub,sample)
    dst=ROOT/f'submissions/{name}.csv';sub.to_csv(dst,index=False)
    info.append(dict(name=name,source=str(path.relative_to(ROOT)),rows=len(sub),phones=sub.phone.nunique(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
(ROOT/'artifacts/reference_submission_manifest.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info),flush=True)
