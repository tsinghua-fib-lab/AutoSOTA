"""Infer an additional checkpoint with the original normalization and masks."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from greysnow_runtime import GreySnowRuntime,ROOT,np,tf,macro_r2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--model-id",required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--kind",choices=["validation","test"],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--rh-feature",action="store_true")
    parser.add_argument("--buoyancy-feature",action="store_true")
    args=parser.parse_args()
    tf.keras.mixed_precision.set_global_policy("float32")
    if args.kind=="validation":
        if not json.loads((args.cache/"manifest.json").read_text()).get("validation_only"):raise ValueError("Expected validation-only cache")
    else:
        manifest=json.loads((args.cache/"input_manifest.json").read_text())
        if manifest["rows"]!=625000:raise ValueError("Unexpected competition test row count")
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    checksum=digest.hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"completed.json").exists():
        old=json.loads((args.output/"completed.json").read_text())
        if old["checkpoint_sha256"]==checksum and old["kind"]==args.kind and old.get('rh_feature',False)==args.rh_feature and old.get('buoyancy_feature',False)==args.buoyancy_feature:
            print('Cached extra checkpoint inference',flush=True);return
    runtime=GreySnowRuntime()
    config=next(m for m in json.loads((ROOT/"configs/champion_models.json").read_text()) if m["id"]==args.model_id)
    if args.buoyancy_feature:
        if not args.rh_feature:raise ValueError('Buoyancy checkpoint includes RH')
        from buoyancy_model import build_physics_model,encode_with_physics
        model,_=build_physics_model(runtime,args.checkpoint,config,warmstart=False)
    elif args.rh_feature:
        from augmented_greysnow import build_rh_model,encode_with_rh
        model,_=build_rh_model(runtime,args.checkpoint,config,warmstart=False)
    else:model=runtime.build_model(args.checkpoint,config["dim"],config["head_dim"])
    names=json.loads((ROOT/"artifacts/champion-validation"/args.model_id/"metrics.json").read_text())["bad_targets"]
    bad=np.asarray([c in names for c in runtime.target_columns])
    x=np.load(args.cache/"encoded.npy",mmap_mode='r');humidity=np.load(args.cache/"humidity.npy",mmap_mode='r')
    rh=np.load(args.cache/'rh.npy',mmap_mode='r') if args.rh_feature else None
    buoyancy=np.load(args.cache/'buoyancy.npy',mmap_mode='r') if args.buoyancy_feature else None
    physical=np.lib.format.open_memmap(args.output/"predictions.npy",mode='w+',dtype=np.float64,shape=(len(x),368))
    normalized=np.lib.format.open_memmap(args.output/"normalized_predictions.npy",mode='w+',dtype=np.float32,shape=(len(x),748))
    @tf.function(reduce_retracing=True)
    def infer(features):return model(features,training=False)
    started=time.monotonic()
    for offset in range(0,len(x),args.batch_size):
        end=min(offset+args.batch_size,len(x));features=x[offset:end]
        if args.buoyancy_feature:features=encode_with_physics(features,rh[offset:end],buoyancy[offset:end])
        elif args.rh_feature:features=encode_with_rh(features,rh[offset:end])
        p=infer(tf.convert_to_tensor(features)).numpy()
        if not np.isfinite(p).all():raise FloatingPointError('Non-finite model predictions')
        normalized[offset:end]=p;physical[offset:end]=runtime.physical_predictions(p,humidity[offset:end,60:90],bad)
        if offset%51200==0:print(f'{args.kind}: {end}/{len(x)}; seconds={time.monotonic()-started:.1f}',flush=True)
    physical.flush();normalized.flush()
    report={"rows":len(x),"kind":args.kind,"checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":checksum,"source_model":args.model_id,"rh_feature":args.rh_feature,"buoyancy_feature":args.buoyancy_feature,"bad_targets":names,"elapsed_seconds":time.monotonic()-started,"test_labels_used":False}
    if args.kind=="validation":
        labels=np.load(ROOT/"artifacts/champion-validation/lr_3e/labels.npy",mmap_mode='r')
        if len(labels)!=len(x):raise ValueError('Validation row order mismatch')
        report["local_full_holdout_r2"],_=macro_r2(labels,physical,runtime.weights)
    (args.output/"completed.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
