import numpy as np
import matplotlib.pyplot as plt

# A Function For Automatically Detecting The Time Range Of CO Occurance
def detect_window_range(window_size, alignments, tail_portion=0.25):
    """
    Automatically detect time ranges of Catastrophic Overfitting (CO) based on alignment drift.

    Compares the mean alignment at the start and end portions of a sliding window.
    A CO window is flagged if their absolute difference exceeds 0.5.

    Args:
        window_size (int): Total length of the sliding window in samples.
        alignments (list[float] or np.ndarray): Alignment values per time step.
        tail_portion (float): Fraction of the window considered "start" and "end"
            for averaging. Defaults to 0.25 (i.e., 25% of window length).

    Returns:
        list[tuple[int, int]]: List of (start, end) index ranges for detected CO windows.
    """
    window_list = [(0 , -1)]
    tail_size = int(window_size * tail_portion)
    capture_next = True
    for start_index in range(len(alignments) - window_size):
        avg_alignment_start = np.mean(alignments[start_index: start_index + tail_size])
        avg_alignment_finish = np.mean(alignments[start_index + window_size - tail_size: start_index + window_size])
        if abs(avg_alignment_start - avg_alignment_finish) >= 0.5:
            if capture_next:
                window_list.append((start_index, start_index + window_size))
                capture_next = False
        else:
            capture_next = True
    return window_list

def plot_loss_and_accuracy(args, training_loss, training_accuracy, attack_evaluation_loss, attack_evaluation_accuracy, benign_evaluation_loss,
                            benign_evaluation_accuracy, fgsm_evaluation_loss, fgsm_evaluation_accuracy, pgd_evaluation_loss, pgd_evaluation_accuracy):
    """
    Plot training/test loss and accuracy curves for multiple evaluation modes.

    Shows:
        - Training vs. test loss for the current attack.
        - Test loss for FGSM, PGD, and benign (clean) evaluation.
        - Training vs. test accuracy for the same settings.

    Saves:
        loss_accuracy_plot.png in the corresponding results folder.

    Args:
        args (Namespace): Holds dataset, model, attack, root_path, seed.
        *_loss (list[float]): Loss history over epochs for each evaluation mode.
        *_accuracy (list[float]): Accuracy history for each evaluation mode.
    """
    figure, axis = plt.subplots(1,2, figsize=(24,12))
    axis[0].plot(np.arange(len(training_loss)) ,training_loss, '-o', label=f'Train Loss {args.attack}')
    axis[0].plot(np.arange(len(attack_evaluation_loss)) ,attack_evaluation_loss, '-o', label=f'Test Loss {args.attack}')
    axis[0].plot(np.arange(len(fgsm_evaluation_loss)) ,fgsm_evaluation_loss, '-o', label='Test Loss FGSM')
    axis[0].plot(np.arange(len(pgd_evaluation_loss)) ,pgd_evaluation_loss, '-o', label='Test Loss PGD')
    axis[0].plot(np.arange(len(benign_evaluation_loss)) ,benign_evaluation_loss, '-o', label='Test Loss Benign')
    axis[0].set_title("Loss vs. Epochs")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Loss")
    axis[0].legend()
    axis[0].grid()
    axis[1].set_title("Accuracy vs. Epochs")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Accuracy")
    axis[1].plot(np.arange(len(training_accuracy)),training_accuracy, '-o', label=f'Train Accuracy {args.attack}')
    axis[1].plot(np.arange(len(attack_evaluation_accuracy)),attack_evaluation_accuracy, '-o', label=f'Test Accuracy {args.attack}')
    axis[1].plot(np.arange(len(fgsm_evaluation_accuracy)),fgsm_evaluation_accuracy, '-o', label='Test Accuracy FGSM')
    axis[1].plot(np.arange(len(pgd_evaluation_accuracy)),pgd_evaluation_accuracy, '-o', label='Test Accuracy PGD')
    axis[1].plot(np.arange(len(benign_evaluation_accuracy)),benign_evaluation_accuracy, '-o', label='Test Accuracy Benign')
    axis[1].legend()
    axis[1].grid(visible=True, which= 'minor', color='k', linestyle='-', alpha=0.4)
    axis[1].grid(visible=True, which= 'major', color='b', linestyle='-', alpha=0.8)
    plt.minorticks_on()
    plt.savefig(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}/loss_accuracy_plot.png")
    plt.show()

def plot_regularizer_values(args, reg_train_values: list, reg_test_values: list):
    """
    Plot regularizer values over training batches for train and test datasets.

    Args:
        args (Namespace): Holds dataset, model, attack, root_path, seed.
        reg_train_values (list[float]): Regularizer values per batch (training set).
        reg_test_values (list[float]): Regularizer values per batch (test set).
    """
    figure, axis = plt.subplots(1,2, figsize=(24,12))
    axis[0].plot(reg_train_values)
    axis[0].set_title(f"{args.attack} Regularizer During Training (Train Dataset)")
    axis[0].set_xlabel("Batch")
    axis[0].set_ylabel("Regularizer Value")
    axis[0].grid()
    
    axis[1].plot(reg_test_values)
    axis[1].set_title(f"{args.attack} Regularizer During Training (Test Dataset)")
    axis[1].set_xlabel("Batch")
    axis[1].set_ylabel("Regularizer Value")
    axis[1].grid()

    plt.savefig(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}/regularizer_plot.png")
    plt.show()



def plot_alpha_per_batch(args, alpha_values: list):
    """
    Plot alpha hyperparameter progression over training batches.

    Args:
        args (Namespace): Holds dataset, model, attack, root_path, seed.
        alpha_values (list[float]): Alpha value per batch.
    """
    plt.figure(figsize=(24,12))
    plt.plot(alpha_values)
    plt.title(f"Alpha During {args.attack} Training Vs. Batch")
    plt.xlabel("Batch")
    plt.ylabel("Alpha")
    plt.grid()
    plt.savefig(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}/alpha_plot.png")
    plt.show()

def plot_alignment(args, alignments: list):
    """
    Plot gradient alignment over training batches.

    Args:
        args (Namespace): Holds dataset, model, attack, root_path, seed.
        alignments (list[float]): Alignment values per batch.
    """
    plt.figure(figsize=(24,12))
    plt.plot(alignments)
    plt.title(f"Alignment During {args.attack} Training Vs. Batch")
    plt.xlabel("Batch")
    plt.ylabel("Alignment")
    plt.grid()
    plt.savefig(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}/alignment_plot.png")
    plt.show()

# Plot FGSM and PGD accuracies in final checkpoint for different epsilons 
def plot_accs_vs_eps(args, fgsm_accs: list, pgd_accs: list, clean_acc: float):
    """
    Plot FGSM and PGD accuracies at final checkpoint vs. epsilon.

    Args:
        args (Namespace): Holds dataset, model, attack, root_path, seed.
        fgsm_accs (list[float]): FGSM accuracy (%) for each epsilon value.
        pgd_accs (list[float]): PGD accuracy (%) for each epsilon value.
        clean_acc (float): Clean accuracy (%), plotted as horizontal dashed line.
    """
    epsilons = list(range(1, 1 + len(fgsm_accs)))  # [1, 2, 3,...]
    
    plt.figure(figsize=(24, 12))
    
    # Plot clean accuracy as horizontal line
    plt.axhline(y=clean_acc, color='k', linestyle='--', linewidth=2, label='Clean')
    
    # Plot adversarial accuracies with markers
    plt.plot(epsilons, fgsm_accs, 'o-', label='FGSM')
    plt.plot(epsilons, pgd_accs, 's-', label='PGD')
    
    # Formatting
    plt.ylim(0-5, 100+5)  # Force 0-100% range
    plt.xticks(epsilons)
    plt.xlabel(r'Attack Strength ($\epsilon \times 255$)')
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Robustness Across Attack Strengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{args.root_path}/Results/{args.dataset}/{args.model}/{args.attack}/plots_{args.seed}/acc_vs_eps.png", bbox_inches='tight')
    plt.show()
