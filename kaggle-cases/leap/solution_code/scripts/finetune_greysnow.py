"""Metric-aligned fine-tuning of a published checkpoint on training shards only."""
import argparse
import json
import math
import os
import time
from pathlib import Path

from greysnow_runtime import GreySnowRuntime,ROOT,DATA,np,tf,macro_r2,load_pickle


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
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
    parser.add_argument("--hr-softclip",action="store_true")
    parser.add_argument("--rh-feature",action="store_true")
    parser.add_argument("--ema-decay",type=float,default=0)
    parser.add_argument("--rh-learning-rate",type=float,default=0)
    parser.add_argument("--balance-months",action="store_true")
    parser.add_argument("--buoyancy-feature",action="store_true")
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
    if args.buoyancy_feature:
        if not args.rh_feature:raise ValueError('Buoyancy augmentation includes RH')
        from buoyancy_model import build_physics_model,encode_with_physics
        model,original_model=build_physics_model(runtime,checkpoint,config,warmstart=True)
    elif args.rh_feature:
        from augmented_greysnow import build_rh_model,encode_with_rh
        model,original_model=build_rh_model(runtime,checkpoint,config,warmstart=True)
    else:model=runtime.build_model(checkpoint,config["dim"],config["head_dim"])
    if args.head_only:
        # Wide prediction head and final prediction layer; retain the backbone.
        for layer in model.layers:layer.trainable=isinstance(layer,tf.keras.layers.Dense) and layer.output_shape[-1]==20 or layer.__class__.__name__ in ["HeadDense","GLUMlp"]
    x=np.load(args.train_cache/"encoded.npy",mmap_mode="r")
    rh=np.load(args.train_cache/"rh.npy",mmap_mode='r') if args.rh_feature else None
    buoyancy=np.load(args.train_cache/'buoyancy.npy',mmap_mode='r') if args.buoyancy_feature else None
    y=np.load(args.train_cache/"targets.npy",mmap_mode="r")
    if len(x)!=manifest["rows"] or len(y)!=len(x):raise ValueError("Training cache dimensions differ")
    additional=None
    if args.additional_cache:
        extra_manifest=json.loads((args.additional_cache/"manifest.json").read_text())
        if extra_manifest.get("validation_only") or not extra_manifest["champion_holdout_excluded"]:raise ValueError("Additional training data contains holdout")
        for path in extra_manifest["shards"]:
            if int(Path(path).stem)<extra_manifest.get("holdout_shards",40):raise ValueError("Holdout in additional data")
        additional=(np.load(args.additional_cache/"encoded.npy",mmap_mode="r"),np.load(args.additional_cache/"targets.npy",mmap_mode="r"))
        extra_rh=np.load(args.additional_cache/"rh.npy",mmap_mode='r') if args.rh_feature else None
        extra_buoyancy=np.load(args.additional_cache/'buoyancy.npy',mmap_mode='r') if args.buoyancy_feature else None
        if not 0<args.additional_fraction<1:raise ValueError("Mixture fraction must lie in (0,1)")
    valx=np.load(args.validation_cache/"encoded.npy",mmap_mode="r")[:args.validation_rows]
    if args.rh_feature:
        valrh=np.load(args.validation_cache/"rh.npy",mmap_mode='r')[:args.validation_rows]
        valbuoyancy=np.load(args.validation_cache/'buoyancy.npy',mmap_mode='r')[:args.validation_rows] if args.buoyancy_feature else None
        probe=np.asarray(valx[:16]);extended=encode_with_physics(probe,valrh[:16],valbuoyancy[:16]) if args.buoyancy_feature else encode_with_rh(probe,valrh[:16])
        before=original_model(probe,training=False).numpy();after=model(extended,training=False).numpy()
        rmse=float(np.sqrt(np.mean((before-after)**2)))
        if rmse>1e-3:raise ValueError('Zero-initialized augmentation changed original predictions')
        write_json(args.output/"augmentation-initialization.json",{'normalized_prediction_rmse':rmse,'zero_initialized_channel':True})
        del original_model
        valx=encode_with_physics(valx,valrh,valbuoyancy) if args.buoyancy_feature else encode_with_rh(valx,valrh)
    humidity=np.load(args.validation_cache/"humidity.npy",mmap_mode="r")[:args.validation_rows]
    labels=np.load(ROOT/"artifacts/champion-validation/lr_3e/labels.npy",mmap_mode="r")[:args.validation_rows]
    bad_names=json.loads((ROOT/"artifacts/champion-validation"/args.model_id/"metrics.json").read_text())["bad_targets"]
    bad=np.asarray([c in bad_names for c in runtime.target_columns])
    active=(runtime.weights!=0).astype(np.float32);active[132:148]=0;active[bad]=0
    mask=tf.constant(active[None])
    schedule=tf.keras.optimizers.schedules.CosineDecay(args.lr,args.steps,alpha=.2)
    optimizer=tf.keras.optimizers.AdamW(learning_rate=schedule,weight_decay=1e-6,global_clipnorm=1)
    optimizer.build(model.trainable_weights)
    rh_kernel=None;rh_optimizer=None
    if args.rh_learning_rate:
        if not args.rh_feature or args.head_only:raise ValueError('A separate RH rate requires a trainable augmented input kernel')
        input_channels=46 if args.buoyancy_feature else 45
        kernels=[v for v in model.trainable_weights if len(v.shape)==2 and v.shape[0]==input_channels]
        if len(kernels)!=1:raise ValueError('Could not identify the augmented input kernel')
        rh_kernel=kernels[0]
        rh_optimizer=tf.keras.optimizers.AdamW(learning_rate=tf.keras.optimizers.schedules.CosineDecay(args.rh_learning_rate,args.steps,alpha=.2),weight_decay=0,global_clipnorm=1)
        rh_optimizer.build([rh_kernel])
    if not 0<=args.ema_decay<1:raise ValueError('EMA decay must be in [0,1)')
    ema_module=tf.Module(name='parameter_ema')
    ema_weights=[]
    if args.ema_decay:
        for i,variable in enumerate(model.trainable_weights):
            shadow=tf.Variable(variable,trainable=False,name=f'ema_{i}')
            setattr(ema_module,f'weight_{i}',shadow);ema_weights.append(shadow)
    anchor=[tf.identity(v) for v in model.trainable_weights] if args.anchor else None
    args.output.mkdir(parents=True,exist_ok=True)
    trackables={'model':model,'optimizer':optimizer,'ema':ema_module}
    if rh_optimizer is not None:trackables['rh_optimizer']=rh_optimizer
    resume_checkpoint=tf.train.Checkpoint(**trackables)
    manager=tf.train.CheckpointManager(resume_checkpoint,str(args.output/"resume"),max_to_keep=2)
    if args.resume and manager.latest_checkpoint:resume_checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    rng=np.random.default_rng(args.seed+int(optimizer.iterations.numpy()))
    def sampler(cache):
        if not args.balance_months:
            rows=json.loads((cache/'manifest.json').read_text())['rows']
            return lambda n:rng.integers(0,rows,size=n)
        months=np.load(cache/'metadata.npy',mmap_mode='r')[:,1]
        groups=[np.flatnonzero(months==month) for month in range(1,13)]
        if any(len(group)==0 for group in groups):raise ValueError('Calendar-balanced sampler requires all months')
        def draw(n):
            chosen=rng.integers(0,12,size=n);indices=np.empty(n,np.int64)
            for month,group in enumerate(groups):
                positions=np.flatnonzero(chosen==month);indices[positions]=group[rng.integers(0,len(group),size=len(positions))]
            return indices
        return draw
    draw=sampler(args.train_cache);extra_draw=sampler(args.additional_cache) if additional is not None else None
    if args.hr_softclip:
        lower=1.1*(np.asarray(load_pickle("leap-sample-submission-stds-ds","y_min_per_col_all")).reshape(-1)-runtime.mean_y)*runtime.stds
        upper=1.1*(np.asarray(load_pickle("leap-sample-submission-stds-ds","y_max_per_col_all")).reshape(-1)-runtime.mean_y)*runtime.stds
        def clip_hr(targets):
            targets=np.where(targets<lower,lower-np.log1p(np.maximum(lower-targets,0)),targets)
            return np.where(targets>upper,upper+np.log1p(np.maximum(targets-upper,0)),targets).astype(np.float32)
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
        rh_gradient=None
        if rh_optimizer is not None:
            index=next(i for i,v in enumerate(model.trainable_weights) if v is rh_kernel)
            channel_mask=tf.one_hot(28,input_channels,dtype=gradients[index].dtype)[:,None]
            if args.buoyancy_feature:channel_mask+=tf.one_hot(29,input_channels,dtype=gradients[index].dtype)[:,None]
            rh_gradient=gradients[index]*channel_mask
            gradients[index]=gradients[index]*(1-channel_mask)
        optimizer.apply_gradients([(g,v) for g,v in zip(gradients,model.trainable_weights) if g is not None])
        if rh_optimizer is not None:rh_optimizer.apply_gradients([(rh_gradient,rh_kernel)])
        if args.ema_decay:
            for shadow,variable in zip(ema_weights,model.trainable_weights):
                shadow.assign(args.ema_decay*shadow+(1-args.ema_decay)*variable)
        return loss
    @tf.function(reduce_retracing=True)
    def infer(features):return tf.cast(model(features,training=False),tf.float32)
    started=time.monotonic();losses=[];history=[];best=-math.inf;bad_evals=0
    if args.resume and (args.output/"history.json").exists():
        history=json.loads((args.output/"history.json").read_text());best=max((h["validation_r2"] for h in history),default=-math.inf)
    for step in range(int(optimizer.iterations.numpy())+1,args.steps+1):
        if (args.output/"stop_requested").exists():
            print("Graceful stop requested",flush=True);break
        indices=draw(args.batch_size)
        features=np.asarray(x[indices]);targets=np.asarray(y[indices])
        batch_rh=np.asarray(rh[indices]).copy() if args.rh_feature else None
        batch_buoyancy=np.asarray(buoyancy[indices]).copy() if args.buoyancy_feature else None
        if additional is not None:
            count=int(round(args.batch_size*args.additional_fraction));extra_indices=extra_draw(count)
            features[:count]=additional[0][extra_indices];targets[:count]=additional[1][extra_indices]
            if args.rh_feature:batch_rh[:count]=extra_rh[extra_indices]
            if args.buoyancy_feature:batch_buoyancy[:count]=extra_buoyancy[extra_indices]
            if args.hr_softclip:targets[:count]=clip_hr(targets[:count])
        if args.buoyancy_feature:features=encode_with_physics(features,batch_rh,batch_buoyancy)
        elif args.rh_feature:features=encode_with_rh(features,batch_rh)
        loss=float(update(tf.convert_to_tensor(features),tf.convert_to_tensor(targets)).numpy())
        if not np.isfinite(loss):raise FloatingPointError("Non-finite loss")
        losses.append(loss)
        if step%20==0 or step==1:
            heartbeat={"step":step,"steps_planned":args.steps,"recent_loss":float(np.mean(losses[-20:])),"elapsed_seconds":time.monotonic()-started,"best_validation_r2":best if math.isfinite(best) else None,"pid":os.getpid(),"updated_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            write_json(args.output/"heartbeat.json",heartbeat)
            print(json.dumps(heartbeat),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            improved=False;raw_weights=None
            for mode in (['raw','ema'] if args.ema_decay else ['raw']):
                if mode=='ema':
                    raw_weights=[tf.identity(v) for v in model.trainable_weights]
                    for variable,shadow in zip(model.trainable_weights,ema_weights):variable.assign(shadow)
                prediction=np.concatenate([infer(tf.convert_to_tensor(valx[start:start+256])).numpy() for start in range(0,len(valx),256)])
                physical=runtime.physical_predictions(prediction,humidity[:,60:90],bad)
                score,_=macro_r2(labels,physical,runtime.weights)
                record={"step":step,"weights_mode":mode,"validation_r2":score,"gain_from_original_checkpoint":score-baseline_score,"train_loss":float(np.mean(losses[-args.eval_every:])),"elapsed_seconds":time.monotonic()-started}
                history.append(record);write_json(args.output/"history.json",history)
                if score>best:
                    best=score;improved=True;model.save_weights(str(args.output/"best.weights.h5"))
                    np.save(args.output/"selection_predictions.npy",physical)
                    write_json(args.output/"best.json",record)
                if mode=='ema':
                    for variable,value in zip(model.trainable_weights,raw_weights):variable.assign(value)
                print('Validation '+json.dumps(record),flush=True)
            bad_evals=0 if improved else bad_evals+1
            manager.save(checkpoint_number=step)
            if args.patience and bad_evals>=args.patience:break
    model.save_weights(str(args.output/"last.weights.h5"))
    write_json(args.output/"completed.json",{"best_validation_r2":best,"baseline_validation_r2":baseline_score,"steps_completed":int(optimizer.iterations.numpy()),"elapsed_seconds":time.monotonic()-started,"test_labels_used":False})
    print('Fine-tuning complete',flush=True)


if __name__=="__main__":main()
