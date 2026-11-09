"""Average best and last checkpoints from one unchanged fine-tuning trajectory."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from greysnow_runtime import ROOT,GreySnowRuntime,macro_r2,np,tf


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--last-fraction',type=float,default=.5);args=parser.parse_args();started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    if not (args.source/'completed.json').exists() or not 0<args.last_fraction<1:raise ValueError('Only a completed single trajectory can be averaged')
    config=json.loads((args.source/'config.json').read_text())
    if config.get('rh_feature') or config.get('buoyancy_feature'):raise ValueError('This initial experiment supports the original architecture only')
    runtime=GreySnowRuntime();model_config=next(m for m in json.loads((ROOT/'configs/champion_models.json').read_text()) if m['id']==config['model_id']);best=args.source/'best.weights.h5';last=args.source/'last.weights.h5';model=runtime.build_model(best,model_config['dim'],model_config['head_dim']);a=model.get_weights();model.load_weights(str(last));b=model.get_weights()
    if len(a)!=len(b) or any(x.shape!=y.shape or x.dtype!=y.dtype for x,y in zip(a,b)):raise ValueError('Checkpoint architectures differ')
    model.set_weights([(1-args.last_fraction)*x+args.last_fraction*y for x,y in zip(a,b)]);model.save_weights(str(args.output/'best.weights.h5'))
    config.update({'steps':0,'checkpoint_operation':'within-trajectory weight average','source_experiment':str(args.source.resolve()),'source_checkpoint_sha256':{'best':digest(best),'last':digest(last)},'last_fraction':args.last_fraction,'rh_feature':False,'buoyancy_feature':False,'batch_size':512});write(args.output/'config.json',config)
    x=np.load(ROOT/'data/finetune-holdout/encoded.npy',mmap_mode='r')[:100000];humidity=np.load(ROOT/'data/finetune-holdout/humidity.npy',mmap_mode='r')[:100000];labels=np.load(ROOT/'artifacts/champion-validation/lr_3e/labels.npy',mmap_mode='r')[:100000];bad_names=json.loads((ROOT/'artifacts/champion-validation'/config['model_id']/'metrics.json').read_text())['bad_targets'];bad=np.array([c in bad_names for c in runtime.target_columns]);prediction=np.empty((100000,368),np.float64)
    @tf.function(reduce_retracing=True)
    def infer(v):return model(v,training=False)
    for start in range(0,len(x),512):
        end=min(start+512,len(x));normalized=infer(tf.convert_to_tensor(x[start:end])).numpy();prediction[start:end]=runtime.physical_predictions(normalized,humidity[start:end,60:90],bad)
        if start%10240==0:write(args.output/'heartbeat.json',{'step':end,'steps_planned':100000,'operation':'checkpoint-average selection inference','elapsed_seconds':time.monotonic()-started});print('Averaged checkpoint selection',end,flush=True)
    if not np.isfinite(prediction).all():raise FloatingPointError('Non-finite averaged predictions')
    score,_=macro_r2(labels,prediction,runtime.weights);old=np.load(ROOT/'artifacts/champion-validation'/config['model_id']/'predictions.npy',mmap_mode='r')[:100000];baseline,_=macro_r2(labels,old,runtime.weights);write(args.output/'best.json',{'step':0,'weights_mode':'average of best and last on one trajectory','validation_r2':score,'gain_from_original_checkpoint':score-baseline});write(args.output/'completed.json',{'best_validation_r2':score,'baseline_validation_r2':baseline,'steps_completed':0,'elapsed_seconds':time.monotonic()-started,'operation':'weight averaging, not additional training','test_labels_used':False});print('Averaged checkpoint score',score,'gain',score-baseline,flush=True)


if __name__=='__main__':main()
