"""Print or execute a relocated inference command for one winning component."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact',required=True,help='Relative prediction artifact from winning_dependency_graph.json')
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    args=ap.parse_args()
    graph=json.loads((ROOT/'winning_dependency_graph.json').read_text())
    node=next(n for n in graph['nodes'] if n['artifact']==args.artifact and n.get('checkpoint'))
    meta=json.loads((ROOT/args.artifact/'metrics.json').read_text())
    ck=ROOT/node['checkpoint']
    old=json.loads((ROOT/'DELIVERY.json').read_text())['runtime_source_root']
    if meta['family']=='team3':
        cmd=[sys.executable,str(ROOT/'scripts/predict_upstream.py'),'--family','team3',
             '--config',meta['config'],'--checkpoint',str(ck),'--fold',str(meta['fold']),
             '--fold-seed',str(meta['fold_seed']),'--packed',str(ROOT/'data/packed'),
             '--mc-folds',str(meta.get('mc_folds',0))]
        # Prefer the historical command for ambiguous TTA settings.
        matches=[]
        for job in json.loads((ROOT/'jobs_full.json').read_text()):
            c=job.get('command',[])
            if '--output' in c:
                out=c[c.index('--output')+1]
                if out in [args.artifact,old+'/'+args.artifact]:matches.append(c)
        mode=meta.get('tta_mode')
        if mode is None:
            for c in matches:
                if '--tta-mode' in c:mode=c[c.index('--tta-mode')+1]
        mode=mode or ('sign_swap' if meta.get('tta') else 'none')
        cmd += ['--tta-mode',mode]
    elif meta['family']=='new_kdl_inspired':
        cmd=[sys.executable,str(ROOT/'scripts/predict_kdl_inspired.py'),'--checkpoint',str(ck),
             '--tta-mode',meta.get('tta_mode') or ('sign' if meta.get('tta') else 'none')]
    else:raise ValueError(meta['family'])
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    cmd+=['--gpu',str(args.gpu),'--output',str(output)]
    print(shlex.join(cmd),flush=True)
    if args.execute:
        for split in ['train','test']:
            assert (ROOT/'data/packed'/f'{split}.npy').is_file(), 'Restore official packed waveforms before inference'
        subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
