import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dp_kfac.config import Config, ScenarioConfig
from dp_kfac.experiment import ExperimentRun
from dp_kfac.models import create_model, get_model_img_size
from dp_kfac.data import get_dataset_loaders, get_dataset_info, get_text_loaders, is_text_dataset
from dp_kfac.trainer import Trainer
from dp_kfac.types import KFACConfig


def run_single_scenario(
    config: Config,
    scenario: ScenarioConfig,
    run: ExperimentRun,
    device: torch.device,
    kfac_config: KFACConfig,
) -> List[Dict[str, Any]]:
    """Run all methods for a single scenario."""
    run.console.print()
    run.console.rule(f"[bold magenta]Scenario: {scenario.name}[/bold magenta]")
    run.log(f"Description: {scenario.description}", "dim")
    run.log(f"Prediction: {scenario.prediction}", "dim")
    run.log(f"Private: {scenario.private_dataset} → Public: {scenario.public_dataset}", "cyan")

    is_text = is_text_dataset(scenario.private_dataset)

    if is_text:
        run.log(f"Loading text datasets...")
        train_loader, test_loader, public_loader, _ = get_text_loaders(
            private_dataset=scenario.private_dataset,
            public_dataset=scenario.public_dataset,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
            num_workers=config.data.num_workers,
            max_samples=config.data.max_samples,
            tokenizer_name="bert-base-uncased",
        )
        num_classes = scenario.num_classes
        img_size = 0
        in_channels = 0
    else:
        in_channels, default_img_size, num_classes = get_dataset_info(scenario.private_dataset)
        img_size = get_model_img_size(config.model.type, default_img_size)
        if scenario.img_size > 0:
            img_size = scenario.img_size
        use_imagenet_norm = config.model.type in ["crossvit", "convnext"]

        run.log(f"Loading datasets...")

        train_loader, test_loader, _ = get_dataset_loaders(
            scenario.private_dataset,
            config.data.batch_size,
            img_size,
            config.data.num_workers,
            use_imagenet_norm,
        )

        public_loader, _, _ = get_dataset_loaders(
            scenario.public_dataset,
            config.data.batch_size,
            img_size,
            config.data.num_workers,
            use_imagenet_norm,
        )

    run.log(f"Creating model: {config.model.type} (classes={num_classes})")
    model = create_model(
        model_type=config.model.type,
        num_classes=num_classes,
        img_size=img_size,
        in_channels=in_channels if not is_text and config.model.type not in ["crossvit", "convnext"] else 3,
        pretrained=config.model.pretrained,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        public_loader=public_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        optimizer_type=config.training.optimizer,
        is_text=is_text,
    )

    scenario_results: List[Dict[str, Any]] = []

    for seed in config.experiment.seeds:
        run.console.print()
        run.console.rule(f"[bold]Seed {seed}[/bold]")

        for epsilon in config.privacy.epsilons:
            run.console.print(f"\n[bold yellow]ε = {epsilon}[/bold yellow]")

            for method in config.methods:
                result = run_method(
                    method=method,
                    trainer=trainer,
                    config=config,
                    kfac_config=kfac_config,
                    epsilon=epsilon,
                    seed=seed,
                    run=run,
                )
                if result:
                    result["scenario"] = scenario.name
                    scenario_results.append(result)

    return scenario_results


def run_method(
    method: str,
    trainer: Trainer,
    config: Config,
    kfac_config: KFACConfig,
    epsilon: float,
    seed: int,
    run: ExperimentRun,
) -> Dict[str, Any] | None:
    """Run a single training method and return results."""

    if method == "baseline" or method == "plain_sgd":
        run.log("Training baseline (Plain SGD)...", "yellow")
        results = trainer.train_baseline(
            epochs=config.training.epochs,
            seed=seed,
        )
        final = results[-1]
        run.log(f"Baseline: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "baseline",
            "seed": seed,
            "epsilon": None,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_sgd":
        run.log(f"Training DP-SGD (ε={epsilon})...", "yellow")
        results = trainer.train_dp_sgd(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
        )
        final = results[-1]
        run.log(f"DP-SGD: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_sgd",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_kfac_public":
        run.log(f"Training DP-KFAC public (A_pub⊗G_pub) (ε={epsilon})...", "yellow")
        results = trainer.train_dp_kfac(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            use_public_data=True,
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-KFAC (Public): Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_kfac_public",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_kfac_noise":
        run.log(f"Training DP-KFAC white noise (A_white⊗G_white) (ε={epsilon})...", "yellow")
        results = trainer.train_dp_kfac(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            use_public_data=False,
            use_pink_noise=False,
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-KFAC (White Noise): Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_kfac_noise",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_kfac_pink":
        run.log(f"Training DP-KFAC pink noise (A_pink⊗G_pink) (ε={epsilon})...", "yellow")
        results = trainer.train_dp_kfac(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            use_public_data=False,
            use_pink_noise=True,
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-KFAC (Pink Noise): Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_kfac_pink",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_adam_bc":
        run.log(f"Training DP-AdamBC (ε={epsilon})...", "yellow")
        results = trainer.train_dp_adam_bc(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            lr=config.baselines.adam_bc_lr,
            eps_root=config.baselines.adam_bc_eps_root,
            betas=tuple(config.baselines.adam_bc_betas),
        )
        final = results[-1]
        run.log(f"DP-AdamBC: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_adam_bc",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_disk":
        run.log(f"Training DiSK (ε={epsilon})...", "yellow")
        results = trainer.train_dp_disk(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            kappa=config.baselines.disk_kappa,
            lr=config.baselines.disk_lr,
        )
        final = results[-1]
        run.log(f"DiSK: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_disk",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_adambc_kfac_synthetic":
        run.log(f"Training DP-AdamBC + Synthetic KFAC (ε={epsilon})...", "yellow")
        results = trainer.train_dp_adambc_kfac_synthetic(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            lr=config.baselines.adambc_kfac_lr or config.baselines.adam_bc_lr,
            eps_root=config.baselines.adam_bc_eps_root,
            betas=tuple(config.baselines.adam_bc_betas),
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-AdamBC+KFC: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_adambc_kfac_synthetic",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    elif method == "dp_kfac_lora":
        run.log(f"Training DP-KFC + LoRA (ε={epsilon})...", "yellow")
        results = trainer.train_dp_kfac_lora(
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            lr=config.training.learning_rate,
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-KFC+LoRA: Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": "dp_kfac_lora",
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    # Hybrid methods using train_dp_kfac_method
    elif method.startswith("dp_kfac_hybrid_"):
        method_display = {
            "dp_kfac_hybrid_pub_pub_noise": "A_pub⊗G_pub_noise",
            "dp_kfac_hybrid_pub_pink_noise": "A_pub⊗G_pink_noise",
            "dp_kfac_hybrid_pink_pub_pub": "A_pink⊗G_pub_pub",
            "dp_kfac_hybrid_pink_pub_noise": "A_pink⊗G_pub_noise",
        }.get(method, method)

        run.log(f"Training DP-KFAC ({method_display}) (ε={epsilon})...", "yellow")
        results = trainer.train_dp_kfac_method(
            method_name=method,
            epochs=config.training.epochs,
            epsilon=epsilon,
            delta=config.privacy.delta,
            max_grad_norm=config.privacy.max_grad_norm,
            seed=seed,
            kfac_config=kfac_config,
        )
        final = results[-1]
        run.log(f"DP-KFAC ({method_display}): Acc={final['accuracy']*100:.2f}%", "green")
        return {
            "method": method,
            "seed": seed,
            "epsilon": epsilon,
            "final_accuracy": final["accuracy"],
            "final_loss": final["test_loss"],
            "history": results,
        }

    else:
        run.log(f"Unknown method: {method}", "red")
        return None


def run_experiment(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    run = ExperimentRun.create(config)

    run.console.print(Panel(f"[bold cyan]{run.run_id}[/bold cyan]", title="Experiment"))
    run.log(f"Config: {config_path}", "dim")
    run.log(f"Output: {run.run_dir}", "dim")

    requested = config.experiment.device
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    run.log(f"Using device: {device}", "cyan")

    kfac_config = KFACConfig(
        damping=config.kfac.damping,
        update_freq=config.kfac.update_frequency,
    )

    all_results: List[Dict[str, Any]] = []

    # Multi-scenario mode
    if config.scenarios:
        run.log(f"Running {len(config.scenarios)} scenarios", "cyan")
        for scenario in config.scenarios:
            scenario_results = run_single_scenario(
                config=config,
                scenario=scenario,
                run=run,
                device=device,
                kfac_config=kfac_config,
            )
            all_results.extend(scenario_results)

    # Single dataset mode (backward compatible)
    else:
        is_text = is_text_dataset(config.data.private_dataset)

        if is_text:
            run.log(f"Loading text datasets: private={config.data.private_dataset}, public={config.data.public_dataset}")
            train_loader, test_loader, public_loader, _ = get_text_loaders(
                private_dataset=config.data.private_dataset,
                public_dataset=config.data.public_dataset,
                batch_size=config.data.batch_size,
                max_length=config.data.max_length,
                num_workers=config.data.num_workers,
                max_samples=config.data.max_samples,
                tokenizer_name="bert-base-uncased",
            )
            num_classes = config.model.num_classes
            img_size = 0
            in_channels = 0
        else:
            in_channels, default_img_size, num_classes = get_dataset_info(config.data.private_dataset)
            img_size = get_model_img_size(config.model.type, default_img_size)
            use_imagenet_norm = config.model.type in ["crossvit", "convnext"]

            run.log(f"Loading datasets: private={config.data.private_dataset}, public={config.data.public_dataset}")

            train_loader, test_loader, _ = get_dataset_loaders(
                config.data.private_dataset,
                config.data.batch_size,
                img_size,
                config.data.num_workers,
                use_imagenet_norm,
            )

            public_loader, _, _ = get_dataset_loaders(
                config.data.public_dataset,
                config.data.batch_size,
                img_size,
                config.data.num_workers,
                use_imagenet_norm,
            )

        run.log(f"Creating model: {config.model.type}")
        model = create_model(
            model_type=config.model.type,
            num_classes=num_classes,
            img_size=img_size,
            in_channels=in_channels if not is_text and config.model.type not in ["crossvit", "convnext"] else 3,
            pretrained=config.model.pretrained,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            public_loader=public_loader,
            device=device,
            learning_rate=config.training.learning_rate,
            optimizer_type=config.training.optimizer,
            is_text=is_text,
        )

        for seed in config.experiment.seeds:
            run.console.print()
            run.console.rule(f"[bold]Seed {seed}[/bold]")

            for epsilon in config.privacy.epsilons:
                run.console.print(f"\n[bold yellow]ε = {epsilon}[/bold yellow]")

                for method in config.methods:
                    result = run_method(
                        method=method,
                        trainer=trainer,
                        config=config,
                        kfac_config=kfac_config,
                        epsilon=epsilon,
                        seed=seed,
                        run=run,
                    )
                    if result:
                        all_results.append(result)

    run.save_results(all_results)
    run.console.print()
    run.log(f"Experiment completed. Results saved to {run.run_dir}", "bold green")

    print_summary(all_results, config)


def print_summary(results: List[Dict[str, Any]], _config: Config) -> None:
    console = Console()

    has_scenarios = any("scenario" in r for r in results)

    table = Table(title="Results Summary", show_header=True, header_style="bold cyan")
    if has_scenarios:
        table.add_column("Scenario", style="magenta")
    table.add_column("Method", style="white")
    table.add_column("Epsilon", style="yellow")
    table.add_column("Seed", style="dim")
    table.add_column("Accuracy", style="green")

    for r in results:
        eps_str = str(r["epsilon"]) if r["epsilon"] is not None else "N/A"
        if has_scenarios:
            table.add_row(
                r.get("scenario", ""),
                r["method"],
                eps_str,
                str(r["seed"]),
                f"{r['final_accuracy']*100:.2f}%"
            )
        else:
            table.add_row(
                r["method"],
                eps_str,
                str(r["seed"]),
                f"{r['final_accuracy']*100:.2f}%"
            )

    console.print()
    console.print(table)


def run_visualization(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    print(f"Running visualization with config: {config_path}")
    print("Visualization not yet implemented")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DP-KFAC: Differentially Private KFAC Optimizer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    exp_parser = subparsers.add_parser("experiment", help="Run training experiment")
    exp_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    vis_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    vis_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.command == "experiment":
        run_experiment(args.config)
    elif args.command == "visualize":
        run_visualization(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
