from convert_mlp import *
from show_heatmap import *
from dataclasses import dataclass
import tyro

@dataclass
class Args:
    exp_path: str = None    # path excluding or after patching_runs
    series_path: str = None  # path excluding or after patching_runs
    skip_convert: bool = False
    skip_vis: bool = False
    do_test: bool = False
    range_50: bool = False

if __name__ == "__main__":
    args = tyro.cli(Args)

    set_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.set_printoptions(sci_mode=False)
    torch.set_default_device(device)

    assert args.exp_path is not None or args.series_path is not None

    paths_to_run = []
    if args.exp_path is not None:
        output_path = "../../../patching_runs" / Path(args.exp_path) / "output.json"
        assert output_path.exists()
        with open(output_path) as f:
            output_dict = json.load(f)
        assert "result_patching_config_global_iteration_2" in output_dict
        paths_to_run = [output_path.parent]
    else:
        series_path = "../../../patching_runs/" / Path(args.series_path)
        for path in glob.glob("exp*", root_dir=series_path):
            output_path = series_path / path / "output.json"
            if output_path.exists():
                with open(output_path) as f:
                    output_dict = json.load(f)
                if "result_patching_config_global_iteration_2" in output_dict:
                    paths_to_run.append(output_path.parent)
    print("all path")
    print(paths_to_run)

    for path in paths_to_run:
        print("current path", path)
        oa_vecs: OptimalQueryBiasVectors = torch.load(path / "oa_vecs.pt", map_location=device, weights_only=False)
        oa_vecs.requires_grad_(False)

        logger = get_logging_function(path, "post_process_logs.txt")

        with open(path / "config.yaml") as f:
            model_name = yaml.safe_load(f)["model_name"]
        task_name = model_name.split("-")[0]
        model = GPT2LMHeadModel.from_pretrained(f"../../../share/saved_models/{model_name}").to(device)
        model.eval()

        orig_model = GPT2LMHeadModel.from_pretrained(f"../../../share/saved_models/{model_name}").to(device)
        orig_model.eval()

        input_len_range = (0, 50) if args.range_50 else (0, 150)
        tokenizer, dataset = get_tokenizer_and_dataset_for_task(task_name, input_len_range, 150, {"period_for_data":3})
        if hasattr(dataset, "BCE"):
            use_BCE = dataset.BCE
        else:
            use_BCE = False
        if not use_BCE:
            collator = customCollator(tokenizer.pad_token_id)
        else:
            collator = customBCECollator(tokenizer.pad_token_id)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collator)
        

        with open(path / "output.json") as f:
            config = json.load(f)["result_patching_config_global_iteration_2"]
        config = convert_keys_to_int(config)
        logger(config)
        
        hooked_model = PruningModelWithHooksForQK(model, config, defaultdict(lambda : 0), hasattr(oa_vecs, "MLPs"), logger)

        if not args.skip_convert:
            if hasattr(oa_vecs, "MLPs"):
                converted_mlp = convert_mlp(hooked_model, oa_vecs, orig_model, dataloader, logger)
            else:
                converted_mlp = convert_mlp_multi_source(hooked_model, oa_vecs, orig_model, dataloader, logger)

            torch.save(converted_mlp, path / "converted_mlp.pt")
            logger("converted mlp saved at", path / "converted_mlp.pt")
        else:
            converted_mlp = torch.load(path / "converted_mlp.pt", map_location=device, weights_only=False)

        if not args.skip_vis:
            if hasattr(oa_vecs, "MLPs"):
                mlp_to_vis = []
                qk_path_to_vis = []
                unembed_path_to_vis = []
                for layer in range(len(config)-1):
                    for head in config[layer]["qk"]:
                        for q, k in config[layer]["qk"][head]:
                            if any(q.endswith(item) for item in mlp_to_vis) or any(k.endswith(item) for item in mlp_to_vis):
                                qk_path_to_vis.append((q, k, layer, head))
                    for mlp_inp in config[layer]["mlp"]:
                        node = f"mlp-{layer}-{mlp_inp}"
                        if node not in converted_mlp:
                            mlp_to_vis.append(node)
                for lm_head_inp in config["lm_head"]:
                    if any(lm_head_inp.endswith(item) for item in mlp_to_vis):
                        unembed_path_to_vis.append(f"lm_head-{lm_head_inp}")
                logger(qk_path_to_vis)
                logger(unembed_path_to_vis)

                cached_data = {}
                if unembed_path_to_vis:
                    for complete_path in unembed_path_to_vis:
                        # complete_path = "lm_head-mlp-0-attn_output-0-2-wte"
                        inp_cache, out_cache = visualize_mlp_logits(complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
                        # k, vocab, in_vocab
                        cached_data[complete_path] = (inp_cache, out_cache)
                
                if qk_path_to_vis:
                    for q_complete_path, k_complete_path, attn_layer_idx, attn_head_idx in qk_path_to_vis:
                        q_inp_cache, k_inp_cache, out_cache = visualize_mlp_attn_products((q_complete_path, k_complete_path), attn_layer_idx, attn_head_idx, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
                        cached_data[(q_complete_path, k_complete_path, attn_layer_idx, attn_head_idx)] = (q_inp_cache, k_inp_cache, out_cache)

                torch.save(cached_data, path / "mlp_input_output.pt")
                logger("inp out mlp saved at", path / "mlp_input_output.pt")
            
            else:
                mlp_to_vis = []
                qk_path_to_vis = []
                unembed_path_to_vis = []
                for layer in range(len(config)-1):
                    for head in config[layer]["qk"]:
                        for q, k in config[layer]["qk"][head]:
                            if any(q.endswith(item) for item in mlp_to_vis) or any(k.endswith(item) for item in mlp_to_vis):
                                qk_path_to_vis.append((q, k, layer, head))
                    if len(config[layer]["mlp"]) > 0:
                        node = f"mlp-{layer}"
                        if node not in converted_mlp:
                            mlp_to_vis.append(node)
                for lm_head_inp in config["lm_head"]:
                    if any(lm_head_inp.endswith(item) for item in mlp_to_vis):
                        unembed_path_to_vis.append(f"lm_head-{lm_head_inp}")
                logger(qk_path_to_vis)
                logger(unembed_path_to_vis)

                cached_data = {}
                if unembed_path_to_vis:
                    for complete_path in unembed_path_to_vis:
                        # complete_path = "lm_head-attn_output-1-2-mlp-0"
                        inp_cache, out_cache = visualize_mlp_logits_multi_source(complete_path, hooked_model, oa_vecs, dataloader, converted_mlp, logger)
                        # k, vocab, in_vocab
                        cached_data[complete_path] = (inp_cache, out_cache)
                
                if qk_path_to_vis:
                    logger("multi source mlp visulzed in QK not yet impletmented. Skip.")
                
                torch.save(cached_data, path / "mlp_input_output.pt")
                logger("inp out mlp saved at", path / "mlp_input_output.pt")

        if args.do_test:
            if len(converted_mlp) > 0:
                hooked_model.set_converted_mlp(converted_mlp)

                total_num = 0
                match_num = 0
                correct_num = 0
                for i, batch in enumerate(dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch.pop("labels")

                    target_logits = orig_model(**batch).logits

                    logits = hooked_model(masks=torch.ones((1, 1), device=device), oa_vecs=oa_vecs, **batch).logits

                    if not use_BCE:
                        shift_target_logits = target_logits[:, :-1]
                        shift_logits = logits[:, :-1]
                        shift_labels = labels[:, 1:]
                        target_predictions = shift_target_logits.argmax(dim=-1)
                        predictions = shift_logits.argmax(dim=-1)

                        match = ((predictions == target_predictions) | (shift_labels == -100)).all(dim=1)
                        match_num += match.sum().item()
                        correct = ((predictions == shift_labels) | (shift_labels == -100)).all(dim=1)
                        correct_num += correct.sum().item()
                        total_num += correct.numel()
                    else:
                        mask = (batch["input_ids"] == tokenizer.pad_token_id) | (batch["input_ids"] == tokenizer.eos_token_id)
                        target_predictions = (target_logits > 0).long()
                        predictions = (logits > 0).long()

                        match = ((predictions == target_predictions).all(dim=-1) | mask).all(dim=1)
                        match_num += match.sum().item()

                        correct = ((predictions == labels).all(dim=-1) | mask).all(dim=1)
                        correct_num += correct.sum().item()
                        total_num += correct.numel()

                    if total_num > 2000:
                        break
                
                acc_match = match_num / total_num
                acc_task = correct_num / total_num
                logger("******** test results when equipped with replaced MLPs ********")
                logger(f"Using primitives: Acc (match): {acc_match:.3f}, Acc (task): {acc_task:.3f}")

            else:
                logger("no MLP is replaced, no need to test")