import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import time 

from latentgeodesics import LatentPointCloud
from latentgeodesics import MultiIndexDataLoader
from latentgeodesics import train_latent_representation

def main(
    args,
    savepath=None,
    experimentname=None,
    startepoch=0,
    loadpath=None,
    warn=True,
    device=None
):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    args_train = args["train"]
    args_data = args.get("data", {})


    # reproducibility
    torch.manual_seed(args_train["seed"])
    torch.backends.cudnn.deterministic = True
    np.random.seed(args_train["seed"])
    torch.backends.cudnn.benchmark = False

    dataset = LatentPointCloud(Path(args_data["load_path"]),
                                    reach = args["data"].get("reach", None),
                                    device=device)
    dataloader = MultiIndexDataLoader(
            dataset,
            batch_size=args_train["batch_size"],
            num_samples=args_data.get("num_samples", len(dataset)),
            shuffle=True
        )
    
    if args["test"]["data"] is not None:
        dataset_test = LatentPointCloud(Path(args["test"]["data"]),
                                            args["data"].get("reach", None),
                                            device=device)
        dataloader_test = MultiIndexDataLoader(
            dataset_test, 
            batch_size=args["test"].get("batch_size", 1024), 
            num_samples=args["test"].get("num_samples", len(dataset_test)),
            shuffle=True
        )
        print("num_test_data: ", dataloader_test.num)   
    else:
        dataloader_test = None
        print("running without test data")

    print("num_train_data: ",dataloader.num)
    # define savepath
    if experimentname is None:
        experimentname = timestr

    total_save_path = Path(savepath) / experimentname
    print("save path ", total_save_path)
    if total_save_path.exists() and startepoch == 0:
        if warn:
            ans = input(
                "\nWARNING: Save path already exists, are you sure you want to "
                "continue? type y for yes: "
            )
            if ans != "y":
                quit()
        else:
            raise FileExistsError(f"Save path {total_save_path} already exists!", total_save_path)
    else:
        try:
            total_save_path.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            raise ValueError(f"can't create save directory at {Path(savepath) / experimentname}. Specify savepath and experimentname") from e
    
    total_save_path = Path(savepath) / experimentname

    network = train_latent_representation(
        dataloader,
        total_save_path,
        args=args,
        eval_every=10,
        save_every=10,
        start_epoch=startepoch,
        load_path=loadpath,
        device=device,
        dataloader_test=dataloader_test
    )
    return network


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to arguments json file.",
    )
    parser.add_argument(
        "-s",
        "--savepath",
        required=False,
        type=str,
        default="runs",
        help="Name of superfolder where experiment results should be saved."
        "Default: /runs/experimentname",
    )

    parser.add_argument(
        "-e",
        "--experimentname",
        required=False,
        type=str,
        help="Name of the subfolder of savepath where experiment results should be saved."
        "Default: timestring",
    )

    parser.add_argument(
        "--startepoch",
        required=False,
        type=int,
        default=0,
        help="Start epoch, use to continue training.",
    )

    parser.add_argument(
        "--loadpath",
        required=False,
        type=str,
        default="",
        help="Load path, use to continue training.",
    )

    parser.add_argument(
        "-d",
        "--device",
        required=False,
        type=str,
        default="cuda",
        help="Device on which to run the script"
    )

    cargs = parser.parse_args()

    with open(cargs.config, "r") as json_data:
        args = json.load(json_data)

    kwargs = vars(cargs)
    kwargs.pop("config")
    logging.getLogger().setLevel(logging.INFO)
    main(args, **kwargs)
