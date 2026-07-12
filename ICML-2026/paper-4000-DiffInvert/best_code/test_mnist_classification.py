# pylint: disable=no-value-for-parameter,missing-module-docstring,missing-function-docstring
import argparse
import tqdm
import yaml
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
import os
import numpy as np
import torch.nn.functional as F

from src import datasets, groups, pretrained_models, \
    energies, group_sampler, group_optimizer, lit_model
from src.utils.ITS.search import InverseTransformationSearch

torch.set_float32_matmul_precision('high')


def main_its(config):
    device = torch.device("cuda:0")
    assert config.dataset == "affNIST", "ITS is only implemented for affNIST."

    # setup datamodule and group
    datamodule = datasets.AffNIST(config)
    # datamodule = datasets.ITSMNIST(config)
    # group = groups.ImageAffine()

    inner_model = pretrained_models.get_resnet18_mnist_classifier_its(device)

    if config.energy == "resnet18_mnist_classifier_logsumexp":
        alternative_energy = None

    elif config.energy == "lielac_vae_ar":
        alternative_energy = energies.LieLACImageVAEEnergy(
            pretrained_models.get_lielac_vae(device, "affine"),
            pretrained_models.get_lielac_ar(device, "affine"),
            groups.ImageAffine()
        ).to(device)

    else:
        raise NotImplementedError(f"Unknown energy type: {config.energy}")

    its = InverseTransformationSearch(
        model=inner_model,
        its_mode=config.its_mode,
        n_samples=config.n_samples,
        n_hypotheses=config.n_hypotheses,
        mc_steps=config.mc_steps,
        change_of_mind=config.change_of_mind,
        en_unique_class_condition=config.en_unique_class_condition,
        alternative_energy=alternative_energy
    )

    # prepare data
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # testing loop
    correct = 0
    total = 0
    xs = []
    for i, (data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        data, target = data.to(device), target.to(device)

        data_transformed, logits = its.infer(data, plot_idx=None)
        xs.append(data_transformed[:, 0])

        if i == 0:
            # draw original and transformed images
            n = min(data.size(0), 8)
            plt.figure(figsize=(12, 6))
            for i in range(n):
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(data[i].cpu().numpy().squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax = plt.subplot(2, n, i + 1 + n)
                plt.imshow(data_transformed[i, 0].cpu().numpy().squeeze(), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.savefig("experiments/its_transformation.png")
            plt.close()

        # pick the class of the leading hypothesis
        leading_logits= logits[:, 0, :]
        preds = leading_logits.argmax(dim=-1)
        batch_accuracy = preds.eq(target).sum().item() / data.size(0)

        correct += preds.eq(target).sum().item()
        total += data.size(0)

        print(f"batch accuracy: {batch_accuracy * 100:.2f}%")

    xs = torch.cat(xs, dim=0)
    xs_32 = F.interpolate(xs, size=(28, 28), mode='bilinear', align_corners=False)
    xs_np = xs_32.cpu().numpy()
    print(xs_np.shape)

    save_dir = f"experiments/repro_mnist_imgs/{config.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    canonicalized_path = os.path.join(save_dir, f"{config.energy}_{config.outer}_canonicalized.npy")
    padded_path = os.path.join(save_dir, "padded_mnist.npy")
    idx = 1
    while os.path.exists(canonicalized_path):
        canonicalized_path = os.path.join(save_dir, f"{config.energy}_{config.outer}_canonicalized_{idx}.npy")
        idx += 1
    np.save(canonicalized_path, xs_np)
    print(f"Saved: {canonicalized_path}")
    if not os.path.exists(padded_path):
        padded_mnist = datasets.PaddedMNIST(config_)
        padded_mnist.prepare_data()
        padded_mnist.setup("test")
        true_loader = padded_mnist.test_dataloader()
        true_mnist = true_loader.dataset[:][0].unsqueeze(1)
        true_32 = F.interpolate(true_mnist, size=(28, 28), mode='bilinear', align_corners=False)
        true_np = true_32.cpu().numpy()
        true_np = true_np.astype(np.float32) / 255.0
        np.save(padded_path, true_np)
        print(f"Saved: {padded_path}")
    else:
        pass

    # final accuracy
    accuracy = correct / total
    print(f'test accuracy: {accuracy * 100:.2f}%.')


def main(config):
    device = torch.device("cuda:0")

    # setup datamodule and group
    if config.dataset == "PaddedMNIST":
        datamodule = datasets.PaddedMNIST(config)
        group = groups.Dummy()

    elif config.dataset == "affNIST":
        datamodule = datasets.AffNIST(config)
        group = groups.ImageAffine()

    elif config.dataset == "homNIST":
        datamodule = datasets.HomNIST(config)
        group = groups.ImageHomography()

    else:
        raise NotImplementedError

    # setup energy
    if config.energy == "none":
        energy = None
        assert config.outer == "none", "If energy is none, outer must be none."

    elif config.energy == "resnet18_mnist_classifier_logsumexp":
        energy = energies.ImageClassifierEnergy(
            pretrained_models.get_resnet18_mnist_classifier(device),
            group
        ).to(device)
        energy = energies.ImageBoundaryEnergy(energy)

    elif config.energy == "lielac_vae_ar":
        name = "affine" if config.dataset == "affNIST" else "homography"
        energy = energies.LieLACImageVAEEnergy(
            pretrained_models.get_lielac_vae(device, name),
            pretrained_models.get_lielac_ar(device, name),
            group
        ).to(device)
        energy = energies.ImageBoundaryEnergy(energy)

    else:
        raise NotImplementedError(f"Unknown energy type: {config.energy}")

    # setup outer model
    if config.outer == "none":
        outer_model = None

    elif config.outer == "kinetic_langevin":
        assert energy is not None, "Energy must be provided for kinetic_langevin."
        outer_model = group_sampler.EnergyKineticLangevinSampler(
            energy,
            temperature=config.temperature,
            step_size=config.step_size,
            steps=config.steps,
            friction=config.friction,
            clip_norm=config.clip_norm,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            dtype=config.dtype
        ).to(device)

    elif config.outer == "focal":
        assert energy is not None, "Energy must be provided for focal."
        outer_model = group_optimizer.FoCalOptimizer(
            energy,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            init_points=config.init_points,
            n_iter=config.n_iter,
            opt_range=(config.opt_range_lower, config.opt_range_upper),
            seed=config.seed,
            verbose=config.verbose
        )

    elif config.outer == "lielac":
        assert energy is not None, "Energy must be provided for LieLAC."
        outer_model = group_optimizer.LieLACOptimizer(
            energy,
            step_size=config.step_size,
            steps=config.steps,
            num_hypothesis=config.num_hypothesis,
            init_scale=config.init_scale,
            verbose=config.verbose
        ).to(device)

    elif config.outer == "diffusion":
        assert energy is not None, "Energy must be provided for diffusion."
        outer_model = group_sampler.EnergyDiffusionSampler(
            energy,
            temperature=config.temperature,
            steps=config.steps,
            noise_min=config.noise_min,
            noise_max=config.noise_max,
            clip_norm=config.clip_norm,
            num_mc=config.num_mc,
            num_hypothesis=config.num_hypothesis,
            dtype=config.dtype,
            verbose=config.verbose,
            temperature_start=getattr(config, 'temperature_start', 1.0),
            temperature_end=getattr(config, 'temperature_end', 1.0),
            use_antithetic=getattr(config, 'use_antithetic', False)
        ).to(device)

    else:
        raise NotImplementedError(f"Unknown outer model type: {config.outer}")

    # setup inner model
    inner_model = pretrained_models.get_resnet18_mnist_classifier(device)

    # setup lit model
    model = lit_model.LitModel(
        inner_model,
        group,
        outer_model,
        task="test/image_classification",
        ensemble=getattr(config, "ensemble", False)
    ).to(device)

    # setup trainer
    trainer = pl.Trainer(
        deterministic=False,  # TODO: change to True and handle cumsum
        devices=1,
        num_nodes=1
    )

    # testing
    trainer.test(model, datamodule=datamodule)

    xs = model.saved_x_transformed_all
    xs = torch.cat(model.saved_x_transformed_all, dim=0)
    xs_32 = F.interpolate(xs, size=(28, 28), mode='bilinear', align_corners=False)
    xs_np = xs_32.cpu().numpy()

    save_dir = f"experiments/repro_mnist_imgs/{config.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    canonicalized_path = os.path.join(save_dir, f"{config.energy}_{config.outer}_canonicalized.npy")
    padded_path = os.path.join(save_dir, "padded_mnist.npy")
    idx = 1
    while os.path.exists(canonicalized_path):
        canonicalized_path = os.path.join(save_dir, f"{config.energy}_{config.outer}_canonicalized_{idx}.npy")
        idx += 1
    np.save(canonicalized_path, xs_np)
    print(f"Saved: {canonicalized_path}")
    if not os.path.exists(padded_path):
        padded_mnist = datasets.PaddedMNIST(config_)
        padded_mnist.prepare_data()
        padded_mnist.setup("test")
        true_loader = padded_mnist.test_dataloader()
        true_mnist = true_loader.dataset[:][0].unsqueeze(1)
        true_32 = F.interpolate(true_mnist, size=(28, 28), mode='bilinear', align_corners=False)
        true_np = true_32.cpu().numpy()
        true_np = true_np.astype(np.float32) / 255.0
        np.save(padded_path, true_np)
        print(f"Saved: {padded_path}")
    else:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    with open(args.config, 'r', encoding="utf-8") as f:
        config_ = EasyDict(yaml.safe_load(f))

    if getattr(config_, "outer") == "its":
        main_its(config_)

    else:
        main(config_)
