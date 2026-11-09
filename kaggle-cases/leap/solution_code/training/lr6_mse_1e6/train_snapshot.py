"""Metric-aligned fine-tuning of a published checkpoint on training shards only."""
import argparse
import json
import math
import os
import time
from pathlib import Path

from greysnow_runtime import GreySnowRuntime,ROOT,DATA,np,tf,macro_r2


def write_json(path,value):
    temp=path.with_name(path.name+".partial")
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-cache",type=Path,required=True)
    parser.add_argument("--additional-cache",type=Path)
    parser.add_argument("--additional-fraction",type=float,default=0.3333333333)
    parser.add_argument("--validation-cache",type=Path,default=ROOT/"data/finetune-holdout")
    parser.add_argument("--model-id",default="hr_5e_best")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--loss",choices=["mse","huber","hybrid"],default="mse")
    parser.add_argument("--lr",type=float,default=5e-6)
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--eval-every",type=int,default=250)
    parser.add_argument("--validation-rows",type=int,default=50000)
    parser.add_argument("--seed",type=int,default=43)
    parser.add_argument("--precision",choices=["float32","mixed_bfloat16"],default="float32")
    parser.add_argument("--head-only",action="store_true")
    parser.add_argument("--noise",type=float,default=0)
    parser.add_argument("--difference-loss",type=float,default=0)
    parser.add_argument("--anchor",type=float,default=0)
    parser.add_argument("--patience",type=int,default=8)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    manifest=json.loads((args.train_cache/"manifest.json").read_text())
    if manifest.get("validation_only") or not manifest["champion_holdout_excluded"]:raise ValueError("Training cache contains held-out validation")
    if args.validation_rows>100000:raise ValueError("The last 100k holdout is reserved for later audit")
    for path in manifest["shards"]:
        if Path(path).stem.isdigit() and int(Path(path).stem)<manifest.get("holdout_shards",40):raise ValueError("Holdout shard in training")
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy(args.precision)
    runtime=GreySnowRuntime()
    config=next(m for m in json.loads((ROOT/"configs/champion_models.json").read_text()) if m["id"]==args.model_id)
    checkpoint=args.checkpoint or DATA/config["dataset"].split("/")[-1]/config["file"]
    model=runtime.build_model(checkpoint,config["dim"],config["head_dim"])
    if args.head_only:
        # Wide prediction head and final prediction layer; retain the backbone.
        for layer in model.layers:layer.trainable=isinstance(layer,tf.keras.layers.Dense) and layer.output_shape[-1]==20 or layer.__class__.__name__ in ["HeadDense","GLUMlp"]
    x=np.load(args.train_cache/"encoded.npy",mmap_mode="r")
    y=np.load(args.train_cache/"targets.npy",mmap_mode="r")
    if len(x)!=manifest["rows"] or len(y)!=len(x):raise ValueError("Training cache dimensions differ")
    additional=None
    if args.additional_cache:
        extra_manifest=json.loads((args.additional_cache/"manifest.json").read_text())
        if extra_manifest.get("validation_only") or not extra_manifest["champion_holdout_excluded"]:raise ValueError("Additional training data contains holdout")
        for path in extra_manifest["shards"]:
            if int(Path(path).stem)<extra_manifest.get("holdout_shards",40):raise ValueError("Holdout in additional data")
        additional=(np.load(args.additional_cache/"encoded.npy",mmap_mode="r"),np.load(args.additional_cache/"targets.npy",mmap_mode="r"))
        if not 0<args.additional_fraction<1:raise ValueError("Mixture fraction must lie in (0,1)")
    valx=np.load(args.validation_cache/"encoded.npy",mmap_mode="r")[:args.validation_rows]
    humidity=np.load(args.validation_cache/"humidity.npy",mmap_mode="r")[:args.validation_rows]
    labels=np.load(ROOT/"artifacts/champion-validation/lr_3e/labels.npy",mmap_mode="r")[:args.validation_rows]
    bad_names=json.loads((ROOT/"artifacts/champion-validation"/args.model_id/"metrics.json").read_text())["bad_targets"]
    bad=np.asarray([c in bad_names for c in runtime.target_columns])
    active=(runtime.weights!=0).astype(np.float32);active[132:148]=0;active[bad]=0
    mask=tf.constant(active[None])
    schedule=tf.keras.optimizers.schedules.CosineDecay(args.lr,args.steps,alpha=.2)
    optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule,weight_decay=1e-6,global_clipnorm=1)
    optimizer.build(model.trainable_weights)
    anchor=[tf.identity(v) for v in model.trainable_weights] if args.anchor else None
    args.output.mkdir(parents=True,exist_ok=True)
    resume_checkpoint=tf.train.Checkpoint(model=model,optimizer=optimizer)
    manager=tf.train.CheckpointManager(resume_checkpoint,str(args.output/"resume"),max_to_keep=2)
    if args.resume and manager.latest_checkpoint:resume_checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    rng=np.random.default_rng(args.seed+int(optimizer.iterations.numpy()))
    source_prediction=np.load(ROOT/"artifacts/champion-validation"/args.model_id/"predictions.npy",mmap_mode="r")[:args.validation_rows]
    baseline_score,_=macro_r2(labels,source_prediction,runtime.weights)
    run_config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write_json(args.output/"config.json",{**run_config,"model_parameters":model.count_params(),"trainable_parameters":int(sum(np.prod(v.shape) for v in model.trainable_weights)),"train_rows":len(x),"source_checkpoint":str(checkpoint),"baseline_validation_r2":baseline_score,"test_labels_used":False})
    @tf.function(reduce_retracing=True)
    def update(features,targets):
        if args.noise:features+=tf.random.normal(tf.shape(features),stddev=args.noise,dtype=features.dtype)
        with tf.GradientTape() as tape:
            # Evaluation mode preserves pretrained normalization statistics.
            pred=tf.cast(model(features,training=False)[:,:368],tf.float32)
            error=(pred-targets)*mask
            if args.loss=="mse":loss=tf.reduce_sum(tf.square(error))/(tf.cast(tf.shape(error)[0],tf.float32)*tf.reduce_sum(mask))
            elif args.loss=="huber":
                absolute=tf.abs(error);loss=tf.reduce_sum(tf.where(absolute<=1,.5*tf.square(error),absolute-.5))/(tf.cast(tf.shape(error)[0],tf.float32)*tf.reduce_sum(mask))
            else:loss=tf.reduce_sum(.8*tf.square(error)+.2*tf.abs(error))/(tf.cast(tf.shape(error)[0],tf.float32)*tf.reduce_sum(mask))
            if args.difference_loss:
                profile=tf.reshape(error[:,:360],[-1,6,60]);loss+=args.difference_loss*tf.reduce_mean(tf.square(profile[:,:,1:]-profile[:,:,:-1]))
            if args.anchor:
                loss+=args.anchor*tf.add_n([tf.reduce_sum(tf.square(tf.cast(v,tf.float32)-a)) for v,a in zip(model.trainable_weights,anchor)])/sum(int(np.prod(v.shape)) for v in model.trainable_weights)
        gradients=tape.gradient(loss,model.trainable_weights)
        optimizer.apply_gradients([(g,v) for g,v in zip(gradients,model.trainable_weights) if g is not None])
        return loss
    @tf.function(reduce_retracing=True)
    def infer(features):return tf.cast(model(features,training=False),tf.float32)
    started=time.monotonic();losses=[];history=[];best=-math.inf;bad_evals=0
    if args.resume and (args.output/"history.json").exists():
        history=json.loads((args.output/"history.json").read_text());best=max((h["validation_r2"] for h in history),default=-math.inf)
    for step in range(int(optimizer.iterations.numpy())+1,args.steps+1):
        if (args.output/"stop_requested").exists():
            print("Graceful stop requested",flush=True);break
        indices=rng.integers(0,len(x),size=args.batch_size)
        features=np.asarray(x[indices]);targets=np.asarray(y[indices])
        if additional is not None:
            count=int(round(args.batch_size*args.additional_fraction));extra_indices=rng.integers(0,len(additional[0]),size=count)
            features[:count]=additional[0][extra_indices];targets[:count]=additional[1][extra_indices]
        loss=float(update(tf.convert_to_tensor(features),tf.convert_to_tensor(targets)).numpy())
        if not np.isfinite(loss):raise FloatingPointError("Non-finite loss")
        losses.append(loss)
        if step%20==0 or step==1:
            heartbeat={"step":step,"steps_planned":args.steps,"recent_loss":float(np.mean(losses[-20:])),"elapsed_seconds":time.monotonic()-started,"best_validation_r2":best if math.isfinite(best) else None,"pid":os.getpid(),"updated_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json(args.output/"heartbeat.json",heartbeat)
            print(json.dumps(heartbeat),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            prediction=np.concatenate([infer(tf.convert_to_tensor(valx[start:start+256])).numpy() for start in range(0,len(valx),256)])
            physical=runtime.physical_predictions(prediction,humidity[:,60:90],bad)
            score,_=macro_r2(labels,physical,runtime.weights)
            record={"step":step,"validation_r2":score,"gain_from_original_checkpoint":score-baseline_score,"train_loss":float(np.mean(losses[-args.eval_every:])),"elapsed_seconds":time.monotonic()-started}
            history.append(record);write_json(args.output/"history.json",history)
            if score>best:
                best=score;bad_evals=0;model.save_weights(str(args.output/"best.weights.h5"))
                np.save(args.output/"selection_predictions.npy",physical)
                write_json(args.output/"best.json",record)
            else:bad_evals+=1
            manager.save(checkpoint_number=step)
            print('Validation '+json.dumps(record),flush=True)
            if args.patience and bad_evals>=args.patience:break
    model.save_weights(str(args.output/"last.weights.h5"))
    write_json(args.output/"completed.json",{"best_validation_r2":best,"baseline_validation_r2":baseline_score,"steps_completed":int(optimizer.iterations.numpy()),"elapsed_seconds":time.monotonic()-started,"test_labels_used":False})
    print('Fine-tuning complete',flush=True)


if __name__=="__main__":main()
