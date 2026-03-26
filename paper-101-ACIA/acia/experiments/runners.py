# acia/experiments/runners.py
import copy

from torch_geometric.data import DataLoader
from acia.datasets.benchmarks import *
from acia.models.training import *
import os
import time


def cmnist(intervention_type='perfect', intervention_strength=1.0):
    print(f"Running Enhanced ColoredMNIST experiment with {intervention_type} intervention")
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_env1 = ColoredMNIST('e1', train=True, intervention_type=intervention_type,
                              intervention_strength=intervention_strength)
    train_env2 = ColoredMNIST('e2', train=True, intervention_type=intervention_type,
                              intervention_strength=intervention_strength)
    test_env1 = ColoredMNIST('e1', train=False,
                             intervention_type=intervention_type,
                             intervention_strength=intervention_strength)
    test_env2 = ColoredMNIST('e2', train=False,
                             intervention_type=intervention_type,
                             intervention_strength=intervention_strength)

    train_loader = DataLoader(ConcatDataset([train_env1, train_env2]), batch_size=32, shuffle=True)
    test_loader = DataLoader(ConcatDataset([test_env1, test_env2]), batch_size=32, shuffle=False)
    model = ImprovedCausalRepresentationNetwork().to(device)
    start_time = time.time()
    history = enhanced_ctrain_model(train_loader, test_loader, model, n_epochs=8)
    training_time = time.time() - start_time
    model.eval()
    metrics = {}
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    metrics['Acc'] = correct / total
    z_L_all, z_H_all, labels, envs = [], [], [], []
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y.cpu())
            envs.append(e.cpu())

    z_L_all = torch.cat(z_L_all)
    z_H_all = torch.cat(z_H_all)
    labels = torch.cat(labels)
    envs = torch.cat(envs)
    metrics['EI'] = compute_environment_independence(z_H_all, labels, envs)
    metrics['LLI'] = compute_low_level_invariance(z_L_all, labels, envs)
    metrics['IR'] = compute_intervention_robustness(model, test_loader, device)

    print(f"Enhanced CMNIST Results ({intervention_type}):")
    print(f"Accuracy: {metrics['Acc']:.4f}")
    print(f"Environment Independence: {metrics['EI']:.4f}")
    print(f"Low-level Invariance: {metrics['LLI']:.4f}")
    print(f"Intervention Robustness: {metrics['IR']:.4f}")
    causal_results = visualize_causal_flow_cmnist(model=model, test_loader=test_loader, device=device, save_dir=f'./results/enhanced_causal_flow_{intervention_type}/')
    metrics['causal_flow_ratio'] = causal_results['attribution_ratio']
    sample_batch = next(iter(train_loader))[0][:1]
    dims = get_dataset_dimensions(model, sample_batch)
    return training_time, dims, history

def rmnist(intervention_type='perfect', intervention_strength=1.0):
    """
    Run the RotatedMNIST experiment with specified intervention type
    """
    print(f"Running RotatedMNIST experiment with {intervention_type} intervention...")
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define rotation angles for environments
    env_angles = {
        'env1': 0,
        'env2': 15,
        'env3': 30,
        'env4': 45,
        'env5': 60
    }

    # Create datasets with specified intervention
    train_data = RotatedMNIST(env_angles, train=True,
                             intervention_type=intervention_type,
                             intervention_strength=intervention_strength)
    test_data = RotatedMNIST(env_angles, train=False,
                            intervention_type=intervention_type,
                            intervention_strength=intervention_strength)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Use the specialized RMNIST model
    model = RMNISTModel().to(device)
    training_time, history = time_training_experiment("RMNIST", model, train_loader, test_loader, 5)

    # history, metrics_tracker = rtrain_model(train_loader, test_loader, model, n_epochs=5)

    # Evaluate metrics
    model.eval()
    metrics = {}

    # Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    metrics['Acc'] = correct / total

    # Calculate other metrics using model outputs
    z_L_all, z_H_all, labels, envs = [], [], [], []
    with torch.no_grad():
        for x, y, e in test_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y.cpu())
            envs.append(e.cpu())

    z_L_all = torch.cat(z_L_all)
    z_H_all = torch.cat(z_H_all)
    labels = torch.cat(labels)
    envs = torch.cat(envs)

    # Environment Independence (EI)
    metrics['EI'] = compute_environment_independence(z_H_all, labels, envs)

    # Low-level Invariance (LLI)
    metrics['LLI'] = compute_low_level_invariance(z_L_all, labels, envs)

    # Intervention Robustness (IR)
    metrics['IR'] = compute_intervention_robustness(model, test_loader, device)

    print(f"RMNIST Results ({intervention_type}):")
    print(f"Accuracy: {metrics['Acc']:.4f}")
    print(f"Environment Independence: {metrics['EI']:.4f}")
    print(f"Low-level Invariance: {metrics['LLI']:.4f}")
    print(f"Intervention Robustness: {metrics['IR']:.4f}")

    # visualize_rmnist_results(model, test_loader, device,
    #                          f"cmnist_{intervention_type}_results.png")

    sample_batch = next(iter(train_loader))[0][:1]
    dims = get_dataset_dimensions(model, sample_batch)

    return training_time, dims, metrics
    # return metrics

def ball_agent(intervention_type='perfect', intervention_strength=1.0):
    print(f"Running Ball Agent experiment with {intervention_type} intervention")
    ball_data = BallAgentDataset(n_balls=4, n_samples=10000, intervention_type=intervention_type, intervention_strength=intervention_strength)

    train_data = BallAgentEnvironment(ball_data, is_train=True)
    test_data = BallAgentEnvironment(ball_data, is_train=False)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BallCausalModel().to(device)

    os.makedirs('results', exist_ok=True)
    training_time = time.time()
    ball_train_model(train_loader, test_loader, model, n_epochs=5)
    training_time = time.time() - training_time
    metrics = compute_metrics(model, test_loader, device)
    formatted_metrics = {'Acc': 1.0 - metrics['position_error'],
        'EI': metrics['env_independence'], 'LLI': metrics['low_level_inv'], 'IR': metrics['intervention_rob']}

    print(f"Ball Agent Results ({intervention_type}):")
    print(f"Accuracy: {formatted_metrics['Acc']:.4f}")
    print(f"Environment Independence: {formatted_metrics['EI']:.4f}")
    print(f"Low-level Invariance: {formatted_metrics['LLI']:.4f}")
    print(f"Intervention Robustness: {formatted_metrics['IR']:.4f}")
    sample_batch = next(iter(train_loader))[0][:1]
    dims = get_dataset_dimensions(model, sample_batch)
    return training_time, dims, formatted_metrics

def camelyon17(intervention_type='perfect', intervention_strength=1.0):
    print(f"Running Camelyon17 experiment with {intervention_type} intervention")
    import time
    start_time = time.time()
    camelyon_data = Camelyon17Simulation(n_hospitals=5, n_samples=2000, size=48, intervention_type=intervention_type, intervention_strength=intervention_strength)
    train_datasets = []
    for hospital_id in range(3):
        hospital_indices = [i for i, h in enumerate(camelyon_data.hospitals) if np.argmax(h) == hospital_id]
        tumor_indices = [i for i in hospital_indices if camelyon_data.labels[i] == 1][:50]
        normal_indices = [i for i in hospital_indices if camelyon_data.labels[i] == 0][:50]
        balanced_indices = tumor_indices + normal_indices
        train_datasets.append(torch.utils.data.Subset(camelyon_data, balanced_indices))

    test_datasets = []
    for hospital_id in range(3, 5):
        hospital_indices = [i for i, h in enumerate(camelyon_data.hospitals) if np.argmax(h) == hospital_id]
        tumor_indices = [i for i in hospital_indices if camelyon_data.labels[i] == 1][:50]
        normal_indices = [i for i in hospital_indices if camelyon_data.labels[i] == 0][:50]
        balanced_indices = tumor_indices + normal_indices
        test_datasets.append(torch.utils.data.Subset(camelyon_data, balanced_indices))

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=32, shuffle=True)
    test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CamelyonModel().to(device)
    trainer = CamelyonTrainer(model)
    best_accuracy = 0
    best_model = None
    patience = 5
    counter = 0
    for epoch in range(20):
        train_metrics, train_acc = trainer.train_epoch(train_loader, device)
        test_loss, test_acc = trainer.evaluate(test_loader, device)
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.2f}%")
            model.load_state_dict(best_model)
            break

        print(f'Epoch {epoch + 1}:')
        print(f'Train - Loss: {train_metrics["total_loss"]:.4f}, '
              f'Pred Loss: {train_metrics["pred_loss"]:.4f}, '
              f'R1: {train_metrics["R1"]:.4f}, '
              f'R2: {train_metrics["R2"]:.4f}, '
              f'Acc: {train_acc:.2f}%')
        print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        visualize_camelyon_results(model, test_loader, epoch, f'results/camelyon_{intervention_type}_epoch_{epoch}.png')

    training_time = time.time() - start_time
    metrics = compute_camelyon_metrics(model, test_loader, device)
    formatted_metrics = {'Acc': metrics['accuracy'], 'EI': metrics['env_independence'],
        'LLI': metrics['low_level_inv'], 'IR': metrics['intervention_rob']}

    sample_batch = next(iter(train_loader))[0][:1]
    dims = get_dataset_dimensions(model, sample_batch)
    print(f"Camelyon17 Results ({intervention_type}):")
    print(f"Accuracy: {formatted_metrics['Acc']:.4f}")
    print(f"Environment Independence: {formatted_metrics['EI']:.4f}")
    print(f"Low-level Invariance: {formatted_metrics['LLI']:.4f}")
    print(f"Intervention Robustness: {formatted_metrics['IR']:.4f}")
    return training_time, dims, formatted_metrics

def run_all_experiments():
    """Run all experiments with both perfect and imperfect interventions"""
    results = {}

    # Perfect interventions
    # results['CMNIST_perfect'] = cmnist(intervention_type='perfect')
    # results['RMNIST_perfect'] = rmnist(intervention_type='perfect')
    # results['BallAgent_perfect'] = ball_agent(intervention_type='perfect')
    results['Camelyon17_perfect'] = camelyon17(intervention_type='perfect')

    # Imperfect interventions
    # results['CMNIST_imperfect'] = cmnist(intervention_type='imperfect', intervention_strength=0.5)
    # results['RMNIST_imperfect'] = rmnist(intervention_type='imperfect', intervention_strength=0.5)
    # results['BallAgent_imperfect'] = ball_agent(intervention_type='imperfect', intervention_strength=0.5)
    results['Camelyon17_imperfect'] = camelyon17(intervention_type='imperfect', intervention_strength=0.5)

    # Save results to CSV
    create_results_table(results)

    return results

def create_results_table(results):
    perfect_data = {
        'Dataset': ['Camelyon17'],
        'Acc': [
            # results['CMNIST_perfect']['Acc'],
            # results['RMNIST_perfect']['Acc'],
            # results['BallAgent_perfect']['Acc'],
            results['Camelyon17_perfect']['Acc']
        ],
        'EI': [
            # results['CMNIST_perfect']['EI'],
            # results['RMNIST_perfect']['EI'],
            # results['BallAgent_perfect']['EI'],
            results['Camelyon17_perfect']['EI']
        ],
        'LLI': [
            # results['CMNIST_perfect']['LLI'],
            # results['RMNIST_perfect']['LLI'],
            # results['BallAgent_perfect']['LLI'],
            results['Camelyon17_perfect']['LLI']
        ],
        'IR': [
            # results['CMNIST_perfect']['IR'],
            # results['RMNIST_perfect']['IR'],
            # results['BallAgent_perfect']['IR'],
            results['Camelyon17_perfect']['IR']
        ]
    }

    imperfect_data = {
        # 'Dataset': ['CMNIST', 'RMNIST', 'Ball Agent', 'Camelyon17'],
        'Dataset': ['Camelyon17'],

        'Acc': [
            # results['CMNIST_imperfect']['Acc'],
            # results['RMNIST_imperfect']['Acc'],
            # results['BallAgent_imperfect']['Acc'],
            results['Camelyon17_imperfect']['Acc']
        ],
        'EI': [
            # results['CMNIST_imperfect']['EI'],
            # results['RMNIST_imperfect']['EI'],
            # results['BallAgent_imperfect']['EI'],
            results['Camelyon17_imperfect']['EI']
        ],
        'LLI': [
            # results['CMNIST_imperfect']['LLI'],
            # results['RMNIST_imperfect']['LLI'],
            # results['BallAgent_imperfect']['LLI'],
            results['Camelyon17_imperfect']['LLI']
        ],
        'IR': [
            # results['CMNIST_imperfect']['IR'],
            # results['RMNIST_imperfect']['IR'],
            # results['BallAgent_imperfect']['IR'],
            results['Camelyon17_imperfect']['IR']
        ]
    }

    perfect_df = pd.DataFrame(perfect_data)
    imperfect_df = pd.DataFrame(imperfect_data)
    perfect_df.to_csv('perfect_intervention_results.csv', index=False)
    imperfect_df.to_csv('imperfect_intervention_results.csv', index=False)
    print("\nResults Table (Perfect Interventions):")
    print(perfect_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nResults Table (Imperfect Interventions):")
    print(imperfect_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

