if __name__ == "__main__":
    # This script generates a bash script to upload all the checkpoints to huggingface
    import json

    path_data = []
    with open('../checkpoints/final_defualt_seed_paths.jsonl', 'r') as f:
        for line in f:
            path_data.append(json.loads(line))

    recipe_display_name = {
        "dolma17": "dolma1_7",
        "no_code": "dolma1_7-no-code", 
        "no_math_no_code": "dolma1_7-no-math-code",
        "no_reddit": "dolma1_7-no-reddit",
        "no_flan": "dolma1_7-no-flan",
        "dolma-v1-6-and-sources-baseline": "dolma1_6plus",
        "c4": "c4",
        "prox_fineweb_pro": "fineweb-pro",
        "fineweb_edu_dedup": "fineweb-edu", 
        "falcon": "falcon",
        "falcon_and_cc": "falcon-and-cc",
        "falcon_and_cc_eli5_oh_top10p": "falcon-and-cc-qc-10p",
        "falcon_and_cc_eli5_oh_top20p": "falcon-and-cc-qc-20p",
        "falcon_and_cc_og_eli5_oh_top10p": "falcon-and-cc-qc-orig-10p",
        "falcon_and_cc_tulu_qc_top10": "falcon-and-cc-qc-tulu-10p",
        "DCLM-baseline": "dclm-baseline",
        "dolma17-75p-DCLM-baseline-25p": "dclm-baseline-25p-dolma1.7-75p",
        "dolma17-50p-DCLM-baseline-50p": "dclm-baseline-50p-dolma1.7-50p", 
        "dolma17-25p-DCLM-baseline-75p": "dclm-baseline-75p-dolma1.7-25p",
        "dclm_ft7percentile_fw2": "dclm-baseline-qc-7p-fw2",
        "dclm_ft7percentile_fw3": "dclm-baseline-qc-7p-fw3",
        "dclm_fw_top10": "dclm-baseline-qc-fw-10p",
        "dclm_fw_top3": "dclm-baseline-qc-fw-3p",
        "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p": "dclm-baseline-qc-10p",
        "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p": "dclm-baseline-qc-20p",
    }

    def model_name_to_repo_name(model_name):
        seed = int(model_name.split('-')[-1])
        assert seed in [6198, 4, 5, 14, 15], f"Unknown seed {seed}"
        assert seed == 6198, f"should be defualt seed {seed}"
        size = model_name.split('-')[-2]
        assert size.endswith('M') or size.endswith('B'), f"Unknown size {size}"
        recipe = '-'.join(model_name.split('-')[:-2])
        if recipe == 'baseline':
            recipe = 'dolma17'
        if recipe in ['DCLM-baseline-25p', 'DCLM-baseline-50p', 'DCLM-baseline-75p']:
            return None
        assert recipe in recipe_display_name, f"Unknown recipe {recipe}"
        recipe = recipe_display_name[recipe]\
        
        return f"DataDecide-{recipe}-{size}"

    repo_names = [model_name_to_repo_name(item['model_name']) for item in path_data]
    recipe_names = set('-'.join(repo_name.split('-')[:-1]) for repo_name in repo_names if repo_name is not None)
    recipe_names = sorted(recipe_names)

    assert len(recipe_names) == 25, f"Expected 25 recipes, got {len(recipe_names)}"

    from collections import defaultdict

    recipe_sizes = defaultdict(set)

    for repo_name in repo_names:
        if repo_name is None:
            continue
        parts = repo_name.split('-')
        recipe = '-'.join(parts[:-1])
        size = parts[-1]
        recipe_sizes[recipe].add(size)

    # Convert sets to sorted lists for better readability
    recipe_sizes = {recipe: sorted(sizes) for recipe, sizes in recipe_sizes.items()}

    for recipe, sizes in recipe_sizes.items():
        assert len(sizes) == 14, f"Expected 14 sizes for {recipe}, got {len(sizes)}"

    import os
    repo_name_2_path = {}
    for item in path_data:
        model_name = item['model_name']
        path = item['checkpoints_location'].replace('weka://oe-eval-default', '/data/input/')
        revison = item['revisions']
        assert len(revison) == 1, f"Multiple revisions {revison}"
        revison = revison[0]
        path = os.path.join(path, revison)
        repo_name = model_name_to_repo_name(model_name)
        repo_name_2_path[repo_name] = path


    with open("final_checkpoint_upload_script.sh", "w") as script_file:
        for repo_name, path in repo_name_2_path.items():
            command = f'python upload_model.py --model_path "{path}" --repo_name {repo_name} --private\n'
            script_file.write(command)


    with open("repo_names.txt", "w") as f:
        for repo_name in repo_names:
            if repo_name is not None:
                f.write(repo_name + "\n")
    print("Script generated: final_checkpoint_upload_script.sh")
    print("Repo names saved to repo_names.txt")
    print("All done!")
    


