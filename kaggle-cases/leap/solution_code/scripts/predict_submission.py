"""Infer official CSV data with original weights and build a weighted submission.

This program requires the actual competition test.csv and sample_submission.csv.
It never reads competition test labels and does not upload anything.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from greysnow_runtime import GreySnowRuntime, ROOT, DATA, np, tf


def prepare(test, output, batch_size):
    import pandas as pd
    runtime = GreySnowRuntime()
    df = pd.read_csv(test, usecols=["sample_id"]+runtime.feature_columns, float_precision="round_trip")
    if df["sample_id"].duplicated().any():raise ValueError("Duplicate test sample IDs")
    profile = df.iloc[:,[df.columns.get_loc(runtime.feature_columns[i]) for i in runtime.profile_indices]].to_numpy(np.float64)
    scalar = df.iloc[:,[df.columns.get_loc(runtime.feature_columns[i]) for i in runtime.scalar_indices]].to_numpy(np.float64)
    if not np.isfinite(profile).all() or not np.isfinite(scalar).all():raise ValueError("Non-finite test inputs")
    ids=np.asarray(df["sample_id"].astype(str),dtype=str)
    np.save(output/"ids.npy",ids)
    np.save(output/"humidity.npy",profile[:,60:240])
    encoded=np.lib.format.open_memmap(output/"encoded.npy",mode="w+",dtype=np.float32,shape=(len(df),1696))
    with tf.device("/CPU:0"):
        @tf.function(reduce_retracing=True)
        def transform_batch(x,s):
            def transform(pair):
                return runtime.normalize_record({"X_col":pair[0],"X_col_not":pair[1],"y":tf.zeros(368,tf.float64)})
            return tf.vectorized_map(transform,(x,s))
        # Check vectorization against the unmodified per-record normalization.
        expected=runtime.normalize_record({"X_col":tf.constant(profile[0]),"X_col_not":tf.constant(scalar[0]),"y":tf.zeros(368,tf.float64)}).numpy()
        actual=transform_batch(tf.constant(profile[:1]),tf.constant(scalar[:1])).numpy()[0]
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
        for offset in range(0,len(df),8192):
            values=transform_batch(tf.constant(profile[offset:offset+8192]),tf.constant(scalar[offset:offset+8192])).numpy()
            if not np.isfinite(values).all():raise FloatingPointError("Non-finite encoded inputs")
            encoded[offset:offset+len(values)]=values
            if offset%81920==0:print(f"Encoded test rows: {offset+len(values)}/{len(df)}",flush=True)
    encoded.flush()
    (output/"input_manifest.json").write_text(json.dumps({"test":str(test.resolve()),"test_size":test.stat().st_size,
                                                        "test_mtime_ns":test.stat().st_mtime_ns,"rows":len(df)},indent=2))


def worker(model_id, cache, batch_size):
    import fcntl
    with (cache/(model_id+".lock")).open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        completed=cache/(model_id+".done.json")
        if completed.exists() and completed.stat().st_mtime_ns >= (cache/"encoded.npy").stat().st_mtime_ns:
            metadata=json.loads(completed.read_text())
            if metadata["rows"]==len(np.load(cache/"encoded.npy",mmap_mode="r")):
                print(f"Cached test inference: {model_id}",flush=True)
                return
        infer_worker(model_id,cache,batch_size)


def infer_worker(model_id, cache, batch_size):
    models=json.loads((ROOT/"configs/champion_models.json").read_text())
    config=next(m for m in models if m["id"]==model_id)
    report=json.loads((ROOT/"artifacts/champion-validation"/model_id/"metrics.json").read_text())
    runtime=GreySnowRuntime()
    checkpoint=DATA/config["dataset"].split("/")[-1]/config["file"]
    model=runtime.build_model(checkpoint,config["dim"],config["head_dim"])
    encoded=np.load(cache/"encoded.npy",mmap_mode="r")
    humidity=np.load(cache/"humidity.npy",mmap_mode="r")
    out=np.lib.format.open_memmap(cache/(model_id+".npy"),mode="w+",dtype=np.float64,shape=(len(encoded),368))
    bad=np.asarray([c in report["bad_targets"] for c in runtime.target_columns])
    @tf.function(reduce_retracing=True)
    def infer(x):return model(x,training=False)
    for start in range(0,len(encoded),batch_size):
        end=min(start+batch_size,len(encoded))
        prediction=infer(tf.convert_to_tensor(encoded[start:end])).numpy()
        out[start:end]=runtime.physical_predictions(prediction,humidity[start:end,60:90],bad)
        if start%(batch_size*200)==0:print(f"{model_id}: {end}/{len(encoded)} rows",flush=True)
    out.flush()
    (cache/(model_id+".done.json")).write_text(json.dumps({"rows":len(encoded),"model":model_id}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test",type=Path)
    parser.add_argument("--sample",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/submissions/submission.csv")
    parser.add_argument("--mode",choices=["original","optimized"],default="original")
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--worker")
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--assemble-only",action="store_true")
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.cache,args.batch_size);return
    if args.test is None or args.sample is None:parser.error("--test and --sample are required")
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cache=args.cache or args.output.parent/"inference-cache"
    cache.mkdir(parents=True,exist_ok=True)
    if not args.assemble_only:prepare(args.test,cache,args.batch_size)
    else:
        manifest=json.loads((cache/"input_manifest.json").read_text())
        if manifest["test"]!=str(args.test.resolve()) or manifest["test_size"]!=args.test.stat().st_size or manifest["test_mtime_ns"]!=args.test.stat().st_mtime_ns:
            raise ValueError("Cached inputs do not match the supplied test CSV")
    runtime=GreySnowRuntime()
    models=json.loads((ROOT/"configs/champion_models.json").read_text())
    def group_worker(pair):
        gpu,group=pair
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu)
        for model in group:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--worker",model["id"],"--cache",str(cache),
                            "--batch-size",str(args.batch_size)],env=env,check=True)
    if not args.assemble_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(group_worker,enumerate([models[i::4] for i in range(4)])))
    ids=np.load(cache/"ids.npy")
    for model in models:
        completion=json.loads((cache/(model["id"]+".done.json")).read_text())
        if completion["rows"]!=len(ids):raise ValueError("Incomplete model predictions")
    result=np.zeros((len(ids),368),np.float64)
    selected="original";configuration=None
    if args.mode=="optimized":
        configuration=json.loads((ROOT/"artifacts/optimized-ensemble/frozen_config.json").read_text())
        selected=configuration["selected"]
    if selected.startswith("group_weights"):
        groups=[("temperature",0,60),("vapor",60,120),("liquid",120,180),("ice",180,240),
                ("wind_u",240,300),("wind_v",300,360),("scalars",360,368)]
        for name,start,end in groups:
            for w,model in zip(configuration["group_weights"][name],models):
                p=np.load(cache/(model["id"]+".npy"),mmap_mode="r")
                result[:,start:end]+=w*p[:,start:end]
    else:
        for model in models:result+=model["weight"]*np.load(cache/(model["id"]+".npy"),mmap_mode="r")
    if selected.endswith("physics"):
        humidity=np.load(cache/"humidity.npy",mmap_mode="r")
        result[:,60:240]=np.maximum(result[:,60:240],-humidity/1200)
    sample=pd.read_csv(args.sample,nrows=1,float_precision="round_trip")
    if sample.columns.tolist()!=["sample_id"]+runtime.target_columns:raise ValueError("Official sample column order differs from archived schema")
    weights=sample[runtime.target_columns].to_numpy(np.float64)[0]
    sample_ids=pd.read_csv(args.sample,usecols=["sample_id"])["sample_id"].astype(str).to_numpy()
    if not np.array_equal(sample_ids,ids):raise ValueError("Official sample IDs or row order differ from test CSV")
    result*=weights
    if not np.isfinite(result).all():raise FloatingPointError("Non-finite submission values")
    temporary=args.output.with_name(args.output.name+".partial")
    if args.output.suffix==".parquet":
        import polars as pl
        pl.DataFrame({"sample_id":ids,**{c:result[:,i] for i,c in enumerate(runtime.target_columns)}}).write_parquet(temporary,compression="zstd")
    else:
        submission=pd.DataFrame(result,columns=runtime.target_columns)
        submission.insert(0,"sample_id",ids)
        submission.to_csv(temporary,index=False,float_format="%.17g")
    temporary.replace(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps({"test":str(args.test.resolve()),
         "sample":str(args.sample.resolve()),"rows":len(ids),"targets":368,"selection":selected,
         "weights_applied_once":True,"uploaded":False},indent=2))
    print(f"Submission ready: {args.output}, rows={len(ids)}, targets=368",flush=True)


if __name__=="__main__":main()
