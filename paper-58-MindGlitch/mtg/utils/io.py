from pathlib import Path


def check_mkdir(path):
    path = Path(path)
    if path.is_file and not path.parents[0].exists():
        path.parents[0].mkdir(parents=True)
    elif not path.exists():
        path.mkdir(parents=True)


## Create output directories
def create_output_dirs(dataset_save_path: str):
    output_dir = Path(f"{dataset_save_path}")
    check_mkdir(output_dir)

    imgs_path = output_dir / "imgs"
    failed_imgs_path = output_dir / "failed_imgs"

    jsons_path = output_dir / "json"
    failed_jsons_path = output_dir / "failed_json"

    correspond_path = output_dir / "correspondences"

    for path in [imgs_path, failed_imgs_path, jsons_path, failed_jsons_path, correspond_path]:
        check_mkdir(path)
    return output_dir, imgs_path, failed_imgs_path, jsons_path, failed_jsons_path, correspond_path
