import json
from visualization.functions import plot_loss_and_accuracy, plot_regularizer_values, plot_alpha_per_batch, plot_alignment, plot_accs_vs_eps

def visualize(args):
    """
    Visualize the results of the training and evaluation.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
    """
    

    training_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/train_metrics.json"
    with open(training_metrics_path, "r") as f:
        training_metrics = json.load(f)

    evaluation_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/evaluation_metrics.json"
    with open(evaluation_metrics_path, "r") as f:
        evaluation_metrics = json.load(f)

    accs_vs_eps_metrics_path = f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/raw_results_{args.seed}/accs_vs_eps_metrics.json"
    with open(accs_vs_eps_metrics_path, "r") as f:
        accs_vs_eps_metrics = json.load(f)


    plot_loss_and_accuracy(args, training_metrics["epoch_metrics"]["loss"], training_metrics["epoch_metrics"]["accuracy"],
                            evaluation_metrics["attack_epoch_metrics"]["loss"], evaluation_metrics["attack_epoch_metrics"]["accuracy"],
                            evaluation_metrics["benign_epoch_metrics"]["loss"], evaluation_metrics["benign_epoch_metrics"]["accuracy"],
                            evaluation_metrics["fgsm_epoch_metrics"]["loss"], evaluation_metrics["fgsm_epoch_metrics"]["accuracy"],
                            evaluation_metrics["pgd_epoch_metrics"]["loss"], evaluation_metrics["pgd_epoch_metrics"]["accuracy"])

    
    if args.attack in ["TRADES", "GradAlign", "ELLE", "NuAT"]:
        plot_regularizer_values(args, training_metrics["regularizer_values"]["batch_train_reg"], evaluation_metrics["regularizer_values"]["batch_test_reg"])
    
    plot_alpha_per_batch(args, training_metrics["alpha_values"]["batch_alpha"])
    
    if args.track_alignment:
        plot_alignment(args, training_metrics["alignment_values"]["batch_alignment"])


    plot_accs_vs_eps(args, accs_vs_eps_metrics["accs_vs_eps_metrics"]["fgsm_acc"], accs_vs_eps_metrics["accs_vs_eps_metrics"]["pgd_acc"],
                      accs_vs_eps_metrics["accs_vs_eps_metrics"]["clean_acc"])
