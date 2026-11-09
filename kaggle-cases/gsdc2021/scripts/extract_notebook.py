"""Export original notebook code; extract %%writefile modules without running cells."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('notebook');p.add_argument('output');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cells=json.loads(Path(a.notebook).read_text())['cells']
main=[]
for i,c in enumerate(cells):
    if c['cell_type'] != 'code':continue
    s=''.join(c['source'])
    if s.startswith('%%writefile '):
        header,body=s.split('\n',1)
        name=header.split(maxsplit=1)[1].strip()
        target=out/name
        if target.parent.resolve()!=out.resolve():raise ValueError(name)
        target.write_text(body)
    else:
        main.append(f'# Original cell {i}\n'+s)
(out/'notebook_cells.py.txt').write_text('\n\n'.join(main))
print('Exported',out)
