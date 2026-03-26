import os
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer, AutoEncoder, EvalActivationBuffer
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.trainers import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import load_dictionary, hf_dataset_to_generator, get_submodule
from dictionary_learning.evaluation import evaluate
from datasets import load_dataset
import argparse
import torch as t
import random
import demo_config
import json

@t.no_grad()
def eval_saes(
    model_name: str,
    ae,
    trainer_cfg,
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = 128
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    # loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    loss_recovered_batch_size = 1
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    # generator = hf_dataset_to_generator("NeelNanda/pile-10k")
    data = iter(load_dataset("NeelNanda/pile-10k", split="train", streaming=True))

    # input_strings = []
    # for i, example in enumerate(generator):
    #     input_strings.append(example)
    #     if i > n_inputs * 5:
    #         break

    eval_results = {}

    dictionary = ae.to(dtype=model.dtype)

    layer = trainer_cfg["trainer"]["layer"]
    submodule = get_submodule(model, layer)

    activation_dim = trainer_cfg["trainer"]["activation_dim"]

    activation_buffer = EvalActivationBuffer(
        data, #iter(input_strings)
        model,
        submodule,
        n_ctxs=1,
        ctx_len=128,
        refresh_batch_size=1,
        out_batch_size=1,
        io=io,
        d_submodule=activation_dim,
        device=device,
        remove_bos=True,
    )

    eval_results = evaluate(
        dictionary,
        activation_buffer,
        context_length,
        loss_recovered_batch_size,
        io=io,
        device=device,
        n_batches=n_batches,
    )

    hyperparameters = {
        "n_inputs": n_inputs,
        "context_length": context_length,
    }
    eval_results["hyperparameters"] = hyperparameters

    print(eval_results)

    output_filename = f"eval_results.json"
    with open(output_filename, "w") as f:
        json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-store-dir", type=str, default="activations")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-160m")
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="temporal")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mu", type=float, default=1e-1)
    parser.add_argument("--temp_alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--sparsemax", action="store_true")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=50_000)
    parser.add_argument("--validate-every-n-steps", type=int, default=10000)
    parser.add_argument("--same-init-for-all-layers", action="store_true")
    parser.add_argument("--norm-init-scale", type=float, default=0.005)
    parser.add_argument("--init-with-transpose", action="store_true")
    parser.add_argument("--remove_bos", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resample-steps", type=int, default=None)
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None, nargs="+")
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=["fineweb", "lmsys_chat"]
    )

    parser.add_argument("--save-path", type=str, help="Checkpoints directory for trained SAEs")
    args = parser.parse_args()

    device = "cuda:0"
    model_name =  args.model # can be any Huggingface model

    model = LanguageModel(
        model_name,
        device_map=device,
    )


    if hasattr(model, "model"):
        submodule = model.model.layers[args.layer] # gemma
    elif hasattr(model, "gpt_neox"):
        submodule = model.gpt_neox.layers[args.layer] 
    else:
        print("model doesnt have .model or .gpt_neox attr, check how to access layers")
        exit(1)
    activation_dim = model.config.hidden_size # 768 # output dimension of the MLP
    # dictionary_size = args.expansion_factor * activation_dim
    dictionary_size = 16384

    # data must be an iterator that outputs strings
    data = iter(load_dataset("monology/pile-uncopyrighted", split="train", streaming=True))

    buffer = ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim, # output dimension of the model component
        n_ctxs=1000,#2e4,  # you can set this higher or lower depending on your available memory
        device=device,
        remove_bos=True, # FIXME: removed bos
        temporal=False,
        ctx_len=256,#128
    )  # buffer will yield batches of tensors of dimension = submodule's output dimension

    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 3e-4,
        "device": device,
        "warmup_steps": 1000,
        "wandb_name": f"{args.model}-L{args.layer}"
        + (f"-{args.run_name}" if args.run_name is not None else ""),
        "steps": args.max_steps,
        "k": args.k,
        # "auxk_alpha": 0.0,
        "layer": args.layer,
        "lm_name": args.model,
    }

    if args.save_checkpoints:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        # desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        # desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints = [0.04, 0.08, 0.2, 0.32, 0.4] ## for 50k steps
        # desired_checkpoints = [0.01, 0.02, 0.05, 0.08, 0.1, 0.5]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(args.max_steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None


    # train the sparse autoencoder (SAE)
    ae = trainSAE(
        data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
        trainer_configs=[trainer_cfg],
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        log_steps=50,
        save_dir=os.path.join(args.save_path, args.run_name),
        steps=args.max_steps,
        save_steps=save_steps,
    )


    ae, trainer_cfg = load_dictionary(f"../../checkpoints/{args.run_name}/trainer_0", device)
    ae.temporal = False
    eval_saes(
        args.model,
        ae,
        trainer_cfg,
        demo_config.eval_num_inputs,
        device,
        overwrite_prev_results=True,
    )

