import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import torch
import numpy as np


def visualize_cmnist_results(model, loader, device, save_path=None):
    """Visualize CMNIST results with a sharper 3D environment visualization"""
    model.eval()
    z_L_all, z_H_all, labels, envs = [], [], [], []

    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y.cpu())
            envs.append(e.cpu())

    z_L_all = torch.cat(z_L_all).numpy()
    z_H_all = torch.cat(z_H_all).numpy()
    labels = torch.cat(labels).numpy()
    envs = torch.cat(envs).numpy()

    # Calculate parity
    parity = np.array(['Even' if y % 2 == 0 else 'Odd' for y in labels])

    # 2D t-SNE for standard plots
    tsne_2d = TSNE(n_components=2, random_state=42)
    z_L_2d = tsne_2d.fit_transform(z_L_all)
    z_H_2d = tsne_2d.fit_transform(z_H_all)

    # 3D t-SNE for environment visualization
    tsne_3d = TSNE(n_components=3, random_state=123, perplexity=50)
    z_H_3d = tsne_3d.fit_transform(z_H_all)

    # Create figure
    fig = plt.figure(figsize=(20, 5), dpi=150)
    gs = fig.add_gridspec(1, 4)

    # Panel 1: Low-level by digit
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(
        x=z_L_2d[:, 0], y=z_L_2d[:, 1],
        hue=labels, palette='tab10',
        legend='brief', ax=ax1, s=30, alpha=0.8
    )
    # ax1.set_title('Low-level Representation by Digit (Label)')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend(title='Digit')

    # Panel 2: High-level by digit
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=labels, palette='tab10',
        legend='brief', ax=ax2, s=30, alpha=0.8
    )
    # ax2.set_title('High-level Representation by Digit (Label)')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(title='Digit')

    # Panel 3: 3D environment visualization
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')

    # Sample a subset of points for clearer visualization
    sample_size = min(500, len(envs))
    indices = np.random.choice(len(envs), sample_size, replace=False)

    # Create a clean, sharp 3D visualization with only colors
    # env_colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange

    for env_val, color, label in zip([0, 1], ['#1f77b4', '#ff7f0e'], ['Env 1', 'Env 2']):
        mask = envs[indices] == env_val
        if np.any(mask):
            # Calculate z-order for sizes
            z_vals = z_H_3d[indices][mask, 2]
            sizes = 20 + 40 * (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())

            # Plot with edge color for better visibility
            ax3.scatter(
                z_H_3d[indices][mask, 0],
                z_H_3d[indices][mask, 1],
                z_H_3d[indices][mask, 2],
                c=color,
                s=sizes,
                alpha=0.7,
                edgecolors='white',
                linewidths=0.5,
                label=f'Env {env_val + 1}'
            )

        # Better viewing angle and labels
    ax3.view_init(elev=30, azim=45)
    # ax3.set_title('High-level by Environment')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_zlabel('t-SNE Dimension 3')
    ax3.legend()

    # Set optimal viewing angle
    ax3.view_init(elev=25, azim=40)

    # Remove gridlines for cleaner look
    ax3.grid(False)

    # Panel 4: High-level by parity
    ax4 = fig.add_subplot(gs[0, 3])
    sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=parity, palette=['#1f77b4', '#ff7f0e'],
        legend='brief', ax=ax4, s=30, alpha=0.8
    )
    # ax4.set_title('High-level Representation by Parity')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(title='Parity')

    plt.tight_layout()
    # plt.suptitle(f'Representation Analysis at Epoch {5}', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def visualize_rmnist_results(model, loader, epoch, save_path=None):
    """Visualize RMNIST results with standard-sized 3D rotation angle panel"""
    device = next(model.parameters()).device
    model.eval()
    z_L, z_H, labels, angles = [], [], [], []

    # Define proper mapping from environment index to rotation angle
    test_angle_map = {2: '30°', 3: '45°', 4: '60°'}

    with torch.no_grad():
        for x, y, e in loader:
            x = x.to(device)
            l, h, _ = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            labels.append(y)
            angles.append([test_angle_map.get(env_idx.item(), f"{env_idx.item()}°") for env_idx in e])

    # Process data for visualization
    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    labels = torch.cat(labels).numpy()
    angles = np.array(sum(angles, []))  # Flatten list of lists

    # Create 2D t-SNE for standard plots
    tsne_2d = TSNE(n_components=2, random_state=42)
    z_L_2d = tsne_2d.fit_transform(z_L)
    z_H_2d = tsne_2d.fit_transform(z_H)

    # Create 3D t-SNE for rotation angle visualization
    tsne_3d = TSNE(n_components=3, random_state=123, perplexity=50)
    z_H_3d = tsne_3d.fit_transform(z_H)

    # Create standard figure with 1x4 layout
    fig = plt.figure(figsize=(20, 5))

    # Panel 1: Low-level by digit
    ax1 = fig.add_subplot(1, 4, 1)
    sns.scatterplot(
        x=z_L_2d[:, 0], y=z_L_2d[:, 1],
        hue=labels, palette='tab10',
        ax=ax1, s=30, alpha=0.8
    )
    # ax1.set_title('Low-level Representation by Digit')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend(title='Digit')

    # Panel 2: High-level by digit
    ax2 = fig.add_subplot(1, 4, 2)
    sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=labels, palette='tab10',
        ax=ax2, s=30, alpha=0.8
    )
    # ax2.set_title('High-level Representation by Digit')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(title='Digit')

    # Panel 3: 3D rotation angle visualization
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')

    # Find unique angles and create color mapping
    unique_angles = np.unique(angles)
    angle_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_angles)))

    # Subsample for clearer visualization
    sample_size = min(300, len(angles))
    indices = np.random.choice(len(angles), sample_size, replace=False)

    # Plot each angle group
    for i, angle in enumerate(unique_angles):
        mask = angles[indices] == angle
        if np.any(mask):
            ax3.scatter(
                z_H_3d[indices][mask, 0],
                z_H_3d[indices][mask, 1],
                z_H_3d[indices][mask, 2],
                c=[angle_colors[i]],
                s=40,
                alpha=0.7,
                label=angle
            )

    # ax3.set_title('High-level by Rotation Angle (3D)')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_zlabel('t-SNE Dimension 3')
    ax3.legend(title='Angle')
    ax3.view_init(elev=30, azim=45)
    ax3.grid(False)

    # Panel 4: High-level by complexity
    ax4 = fig.add_subplot(1, 4, 4)
    complexity_categories = ["Simple" if d in [0, 1, 7] else
                             "Medium" if d in [2, 3, 5] else
                             "Complex" for d in labels]

    sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=complexity_categories, palette='Dark2',
        ax=ax4, s=30, alpha=0.8
    )
    # ax4.set_title('High-level by Digit Complexity')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(title='Complexity')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def visualize_ball_agent_results(model, loader, epoch, save_path=None):
    """Generate visualization for Ball Agent results with 3D intervention panel"""
    model.eval()
    z_L, z_H, positions, interventions, preds = [], [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            l, h, pos_pred = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            positions.append(y.cpu())
            interventions.append(e.cpu())
            preds.append(pos_pred.cpu())

    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    positions = torch.cat(positions).numpy()
    interventions = torch.cat(interventions).numpy()
    preds = torch.cat(preds).numpy()

    # Calculate prediction errors
    errors = np.mean(np.square(preds - positions), axis=1)
    error_quartiles = pd.qcut(errors, 4, labels=["Low", "Medium-Low", "Medium-High", "High"])

    # Convert interventions to categorical
    int_counts = np.sum(interventions, axis=1)
    int_categories = pd.cut(int_counts, bins=[0, 1, 2, 3, 8],
                            labels=["None", "Single", "Double", "Multiple"])

    # Create 2D t-SNE for standard plots
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    z_L_2d = tsne_2d.fit_transform(z_L)
    z_H_2d = tsne_2d.fit_transform(z_H)

    # Create 3D t-SNE for intervention visualization
    tsne_3d = TSNE(n_components=3, random_state=123, perplexity=30)
    z_H_3d = tsne_3d.fit_transform(z_H)

    # Create dataframe for 2D plotting
    df = pd.DataFrame({
        'x_L': z_L_2d[:, 0], 'y_L': z_L_2d[:, 1],
        'x_H': z_H_2d[:, 0], 'y_H': z_H_2d[:, 1],
        'ball1_x': positions[:, 0], 'ball1_y': positions[:, 1],
        'intervention': int_categories,
        'error': error_quartiles
    })

    # Create figure
    fig = plt.figure(figsize=(20, 5))

    # Plot 1: Low-level representation
    ax1 = fig.add_subplot(1, 4, 1)
    scatter1 = sns.scatterplot(
        x=z_L_2d[:, 0], y=z_L_2d[:, 1],
        hue=positions[:, 0], palette='viridis',
        ax=ax1, s=30, alpha=0.8, legend=False
    )
    # ax1.set_title('Low-level Representation')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')

    # Plot 2: High-level representation
    ax2 = fig.add_subplot(1, 4, 2)
    scatter2 = sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=positions[:, 0], palette='viridis',
        ax=ax2, s=30, alpha=0.8, legend=False
    )
    # ax2.set_title('High-level Representation')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')

    # Add colorbar for first two plots
    # cbar_ax = fig.add_axes([0.46, 0.15, 0.01, 0.7])
    # cbar = fig.colorbar(scatter1.collections[0], cax=cbar_ax)
    # cbar.set_label('Ball 1 X-Position')

    # Plot 3: 3D intervention visualization
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')

    # Sample subset for clearer visualization
    sample_size = min(300, len(int_categories))
    indices = np.random.choice(len(int_categories), sample_size, replace=False)

    # Color mapping for intervention categories
    intervention_colors = {
        "None": "#1f77b4",  # Blue
        "Single": "#ff7f0e",  # Orange
        "Double": "#2ca02c",  # Green
        "Multiple": "#d62728"  # Red
    }

    # Plot each intervention category
    for category in ["None", "Single", "Double", "Multiple"]:
        mask = np.array(int_categories)[indices] == category
        if np.any(mask):
            ax3.scatter(
                z_H_3d[indices][mask, 0],
                z_H_3d[indices][mask, 1],
                z_H_3d[indices][mask, 2],
                c=intervention_colors[category],
                s=40,
                alpha=0.7,
                label=category
            )

    # ax3.set_title('High-level by Intervention (3D)')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_zlabel('t-SNE Dimension 3')
    ax3.legend(title='Interventions')
    ax3.view_init(elev=30, azim=45)
    ax3.grid(False)

    # Plot 4: High-level by prediction error
    ax4 = fig.add_subplot(1, 4, 4)
    error_palette = {"Low": "#1a9850", "Medium-Low": "#91cf60",
                     "Medium-High": "#fc8d59", "High": "#d73027"}

    for error_level in ["Low", "Medium-Low", "Medium-High", "High"]:
        mask = df['error'] == error_level
        ax4.scatter(
            df.loc[mask, 'x_H'],
            df.loc[mask, 'y_H'],
            c=error_palette[error_level],
            s=30,
            alpha=0.7,
            label=error_level
        )

    # ax4.set_title('High-level by Prediction Error')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(title='Error Level')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def visualize_camelyon_results(model, loader, epoch, save_path=None):
    model.eval()
    z_L, z_H, labels, hospitals = [], [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            l, h, _ = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            labels.append(y.cpu())
            hospitals.append(e.argmax(dim=1).cpu())
    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    labels = torch.cat(labels).numpy()
    hospitals = torch.cat(hospitals).numpy()
    model.train()
    uncertainties = []
    for x, _, _ in loader:
        x = x.to(device)
        preds = []
        for _ in range(5):
            _, _, logits = model(x)
            preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        preds = np.stack(preds)
        uncertainties.append(np.std(preds, axis=0).mean(axis=1))
    uncertainties = np.concatenate(uncertainties)
    sample_size = min(200, len(labels))
    indices = np.random.choice(len(labels), sample_size, replace=False)
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    z_L_2d = tsne_2d.fit_transform(z_L[indices])
    z_H_2d = tsne_2d.fit_transform(z_H[indices])
    tsne_3d = TSNE(n_components=3, random_state=123, perplexity=30)
    z_H_3d = tsne_3d.fit_transform(z_H[indices])
    subsampled_labels = labels[indices]
    subsampled_hospitals = hospitals[indices]
    subsampled_uncertainties = uncertainties[indices]
    uncertainty_categories = pd.qcut(subsampled_uncertainties, 4, labels=["Low", "Medium-Low", "Medium-High", "High"])

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(1, 4, 1)
    tumor_palette = {0: '#1f77b4', 1: '#d62728'}
    for label_val, color in tumor_palette.items():
        mask = subsampled_labels == label_val
        if np.any(mask):
            ax1.scatter(z_L_2d[mask, 0], z_L_2d[mask, 1],
                c=color, s=50, alpha=0.8,
                label='Tumor' if label_val == 1 else 'Normal')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend(title='Tissue')

    ax2 = fig.add_subplot(1, 4, 2)
    for label_val, color in tumor_palette.items():
        mask = subsampled_labels == label_val
        if np.any(mask):
            ax2.scatter(
                z_H_2d[mask, 0], z_H_2d[mask, 1],
                c=color, s=50, alpha=0.8,
                label='Tumor' if label_val == 1 else 'Normal')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(title='Tissue')

    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    hospital_colors = plt.cm.Set2(np.linspace(0, 1, 5))

    for hospital in range(5):
        mask = subsampled_hospitals == hospital
        if np.any(mask):
            ax3.scatter(z_H_3d[mask, 0], z_H_3d[mask, 1], z_H_3d[mask, 2],
                c=[hospital_colors[hospital]], s=60, alpha=0.7,
                label=f'H{hospital}')

    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_zlabel('t-SNE Dimension 3')
    ax3.legend(title='Hospital')
    ax3.view_init(elev=25, azim=40)
    ax3.grid(False)

    ax4 = fig.add_subplot(1, 4, 4)
    uncertainty_palette = {"Low": "#1a9850", "Medium-Low": "#91cf60", "Medium-High": "#fc8d59", "High": "#d73027"}
    uncertainty_samples = min(100, len(uncertainty_categories))
    uncertainty_indices = np.random.choice(len(uncertainty_categories), uncertainty_samples, replace=False)

    for level in ["Low", "Medium-Low", "Medium-High", "High"]:
        mask = np.array(uncertainty_categories)[uncertainty_indices] == level
        if np.any(mask):
            ax4.scatter(z_H_2d[uncertainty_indices][mask, 0],
                z_H_2d[uncertainty_indices][mask, 1],
                c=uncertainty_palette[level], s=60, alpha=0.7,
                label=level)

    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(title='Uncertainty')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def visualize_results(model, dataloader, epoch, save_path=None):
    device = next(model.parameters()).device
    model.eval()
    all_Z_L, all_Z_H, all_Y, all_E = [], [], [], []
    with torch.no_grad():
        for x, y, e in dataloader:
            x = x.to(device)
            z_L, z_H, _ = model(x)
            all_Z_L.append(z_L.cpu())
            all_Z_H.append(z_H.cpu())
            all_Y.append(y)
            all_E.append(e)
    Z_L = torch.cat(all_Z_L, 0).numpy()
    Z_H = torch.cat(all_Z_H, 0).numpy()
    Y = torch.cat(all_Y, 0).numpy()
    E = torch.cat(all_E, 0).numpy()
    tsne = TSNE(n_components=2, random_state=42)
    Z_L_2d = tsne.fit_transform(Z_L)
    Z_H_2d = tsne.fit_transform(Z_H)
    parity = ['Even' if y % 2 == 0 else 'Odd' for y in Y]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_L_2d[:, 0],
        'y': Z_L_2d[:, 1],
        'Digit': Y}),
        x='x', y='y', hue='Digit', ax=axes[0])
    axes[0].set_title('Low-level Representation by Digit (Label)')
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Digit': Y}),
        x='x', y='y', hue='Digit', ax=axes[1])
    axes[1].set_title('High-level Representation by Digit (Label)')
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Environment': ['Env 1' if e == 0 else 'Env 2' for e in E]}),
        x='x', y='y', hue='Environment', ax=axes[2])
    axes[2].set_title('High-level Representation by Environment')
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Parity': parity}),
        x='x', y='y', hue='Parity', ax=axes[3])
    axes[3].set_title('High-level Representation by Parity')
    plt.suptitle(f'Representation Analysis at Epoch {epoch}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_results_cmnist(model, dataloader, epoch, save_path=None):
    device = next(model.parameters()).device
    model.eval()
    all_Z_L, all_Z_H, all_Y, all_E = [], [], [], []
    with torch.no_grad():
        for x, y, e in dataloader:
            x = x.to(device)
            z_L, z_H, _ = model(x)
            all_Z_L.append(z_L.cpu())
            all_Z_H.append(z_H.cpu())
            all_Y.append(y)
            all_E.append(e)

    Z_L = torch.cat(all_Z_L, 0).numpy()
    Z_H = torch.cat(all_Z_H, 0).numpy()
    Y = torch.cat(all_Y, 0).numpy()
    E = torch.cat(all_E, 0).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    Z_L_2d = tsne.fit_transform(Z_L)
    Z_H_2d = tsne.fit_transform(Z_H)
    parity = ['Even' if y % 2 == 0 else 'Odd' for y in Y]

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.rcParams.update({'font.size': 12})

    # Row 1: By Digit
    sns.scatterplot(data=pd.DataFrame({'x': Z_L_2d[:, 0], 'y': Z_L_2d[:, 1], 'Digit': Y}),
                    x='x', y='y', hue='Digit', ax=axes[0, 0], palette='tab10', legend='full')
    axes[0, 0].set_title('Low-level by (label)', fontsize=14)
    axes[0, 0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 0].set_ylabel('t-SNE Dimension 2', fontsize=12)

    sns.scatterplot(data=pd.DataFrame({'x': Z_H_2d[:, 0], 'y': Z_H_2d[:, 1], 'Digit': Y}),
                    x='x', y='y', hue='Digit', ax=axes[0, 1], palette='tab10', legend='full')
    axes[0, 1].set_title('High-level by Digit (label)', fontsize=14)
    axes[0, 1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 1].set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Row 1: By Environment
    sns.scatterplot(data=pd.DataFrame({'x': Z_L_2d[:, 0], 'y': Z_L_2d[:, 1],
                                       'Environment': ['Env 1 (Even→Red)' if e == 0 else 'Env 2 (Even→Green)' for e in
                                                       E]}),
                    x='x', y='y', hue='Environment', ax=axes[0, 2], legend='full')
    axes[0, 2].set_title('Low-level by Environment', fontsize=14)
    axes[0, 2].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 2].set_ylabel('t-SNE Dimension 2', fontsize=12)

    sns.scatterplot(data=pd.DataFrame({'x': Z_H_2d[:, 0], 'y': Z_H_2d[:, 1],
                                       'Environment': ['Env 1 (Even→Red)' if e == 0 else 'Env 2 (Even→Green)' for e in
                                                       E]}),
                    x='x', y='y', hue='Environment', ax=axes[0, 3], legend='full')
    axes[0, 3].set_title('High-level by Environment', fontsize=14)
    axes[0, 3].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0, 3].set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Row 2: By Parity
    sns.scatterplot(data=pd.DataFrame({'x': Z_L_2d[:, 0], 'y': Z_L_2d[:, 1], 'Parity': parity}),
                    x='x', y='y', hue='Parity', ax=axes[1, 0], legend='full')
    axes[1, 0].set_title('Low-level by Parity', fontsize=14)
    axes[1, 0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 0].set_ylabel('t-SNE Dimension 2', fontsize=12)

    sns.scatterplot(data=pd.DataFrame({'x': Z_H_2d[:, 0], 'y': Z_H_2d[:, 1], 'Parity': parity}),
                    x='x', y='y', hue='Parity', ax=axes[1, 1], legend='full')
    axes[1, 1].set_title('High-level by Parity', fontsize=14)
    axes[1, 1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 1].set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Row 2: Combined views
    for digit in range(10):
        mask = Y == digit
        axes[1, 2].scatter(Z_L_2d[mask, 0], Z_L_2d[mask, 1],
                           c=['red' if e == 0 else 'blue' for e in E[mask]],
                           alpha=0.7, label=f'Digit {digit}')
    axes[1, 2].set_title('Low-level: Digit-Environment Interaction', fontsize=14)
    axes[1, 2].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 2].set_ylabel('t-SNE Dimension 2', fontsize=12)

    for digit in range(10):
        mask = Y == digit
        axes[1, 3].scatter(Z_H_2d[mask, 0], Z_H_2d[mask, 1],
                           c=['red' if e == 0 else 'blue' for e in E[mask]],
                           alpha=0.7, label=f'Digit {digit}')
    axes[1, 3].set_title('High-level: Digit-Environment Interaction', fontsize=14)
    axes[1, 3].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1, 3].set_ylabel('t-SNE Dimension 2', fontsize=12)

    plt.tight_layout()
    plt.suptitle(f'Causal Representation Analysis at Epoch {epoch}', fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def visualize_ball_agent_results(model, loader, epoch, save_path=None):
    """Generate visualization for Ball Agent results with 3D intervention panel"""
    model.eval()
    z_L, z_H, positions, interventions, preds = [], [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            l, h, pos_pred = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            positions.append(y.cpu())
            interventions.append(e.cpu())
            preds.append(pos_pred.cpu())

    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    positions = torch.cat(positions).numpy()
    interventions = torch.cat(interventions).numpy()
    preds = torch.cat(preds).numpy()

    # Calculate prediction errors
    errors = np.mean(np.square(preds - positions), axis=1)
    error_quartiles = pd.qcut(errors, 4, labels=["Low", "Medium-Low", "Medium-High", "High"])

    # Convert interventions to categorical
    int_counts = np.sum(interventions, axis=1)
    int_categories = pd.cut(int_counts, bins=[0, 1, 2, 3, 8],
                            labels=["None", "Single", "Double", "Multiple"])

    # Create 2D t-SNE for standard plots
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
    z_L_2d = tsne_2d.fit_transform(z_L)
    z_H_2d = tsne_2d.fit_transform(z_H)

    # Create 3D t-SNE for intervention visualization
    tsne_3d = TSNE(n_components=3, random_state=123, perplexity=30)
    z_H_3d = tsne_3d.fit_transform(z_H)

    # Create dataframe for 2D plotting
    df = pd.DataFrame({
        'x_L': z_L_2d[:, 0], 'y_L': z_L_2d[:, 1],
        'x_H': z_H_2d[:, 0], 'y_H': z_H_2d[:, 1],
        'ball1_x': positions[:, 0], 'ball1_y': positions[:, 1],
        'intervention': int_categories,
        'error': error_quartiles
    })

    # Create figure
    fig = plt.figure(figsize=(20, 5))

    # Plot 1: Low-level representation
    ax1 = fig.add_subplot(1, 4, 1)
    scatter1 = sns.scatterplot(
        x=z_L_2d[:, 0], y=z_L_2d[:, 1],
        hue=positions[:, 0], palette='viridis',
        ax=ax1, s=30, alpha=0.8, legend=False
    )
    # ax1.set_title('Low-level Representation')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')

    # Plot 2: High-level representation
    ax2 = fig.add_subplot(1, 4, 2)
    scatter2 = sns.scatterplot(
        x=z_H_2d[:, 0], y=z_H_2d[:, 1],
        hue=positions[:, 0], palette='viridis',
        ax=ax2, s=30, alpha=0.8, legend=False
    )
    # ax2.set_title('High-level Representation')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')

    # Add colorbar for first two plots
    cbar_ax = fig.add_axes([0.46, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(scatter1.collections[0], cax=cbar_ax)
    cbar.set_label('Ball 1 X-Position')

    # Plot 3: 3D intervention visualization
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')

    # Sample subset for clearer visualization
    sample_size = min(300, len(int_categories))
    indices = np.random.choice(len(int_categories), sample_size, replace=False)

    # Color mapping for intervention categories
    intervention_colors = {
        "None": "#1f77b4",  # Blue
        "Single": "#ff7f0e",  # Orange
        "Double": "#2ca02c",  # Green
        "Multiple": "#d62728"  # Red
    }

    # Plot each intervention category
    for category in ["None", "Single", "Double", "Multiple"]:
        mask = np.array(int_categories)[indices] == category
        if np.any(mask):
            ax3.scatter(
                z_H_3d[indices][mask, 0],
                z_H_3d[indices][mask, 1],
                z_H_3d[indices][mask, 2],
                c=intervention_colors[category],
                s=40,
                alpha=0.7,
                label=category
            )

    # ax3.set_title('High-level by Intervention (3D)')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.set_zlabel('t-SNE Dimension 3')
    ax3.legend(title='Interventions')
    ax3.view_init(elev=30, azim=45)
    ax3.grid(False)

    # Plot 4: High-level by prediction error
    ax4 = fig.add_subplot(1, 4, 4)
    error_palette = {"Low": "#1a9850", "Medium-Low": "#91cf60",
                     "Medium-High": "#fc8d59", "High": "#d73027"}

    for error_level in ["Low", "Medium-Low", "Medium-High", "High"]:
        mask = df['error'] == error_level
        ax4.scatter(
            df.loc[mask, 'x_H'],
            df.loc[mask, 'y_H'],
            c=error_palette[error_level],
            s=30,
            alpha=0.7,
            label=error_level
        )

    # ax4.set_title('High-level by Prediction Error')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.legend(title='Error Level')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

def visualize_camelyon_results(model, loader, epoch, save_path=None):
    """Generate visualization for Camelyon17 results"""
    model.eval()
    z_L, z_H, labels, hospitals = [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            l, h, _ = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            labels.append(y.cpu())
            hospitals.append(e.argmax(dim=1).cpu())

    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    labels = torch.cat(labels).numpy()
    hospitals = torch.cat(hospitals).numpy()

    # Calculate uncertainty using Monte Carlo dropout
    model.train()  # Enable dropout
    uncertainties = []
    for x, _, _ in loader:
        x = x.to(device)
        # Run multiple forward passes
        preds = []
        for _ in range(5):  # Monte Carlo dropout
            _, _, logits = model(x)
            preds.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

        preds = np.stack(preds)
        uncertainties.append(np.std(preds, axis=0).mean(axis=1))
    uncertainties = np.concatenate(uncertainties)

    # Categorize uncertainty
    uncertainty_categories = pd.qcut(uncertainties, 4, labels=["Low", "Medium-Low", "Medium-High", "High"])

    # Create TSNE embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_L_2d = tsne.fit_transform(z_L)
    z_H_2d = tsne.fit_transform(z_H)

    # Create dataframe for plotting
    df = pd.DataFrame({
        'x_L': z_L_2d[:, 0], 'y_L': z_L_2d[:, 1],
        'x_H': z_H_2d[:, 0], 'y_H': z_H_2d[:, 1],
        'tumor': labels.astype(int),
        'hospital': hospitals.astype(int),
        'uncertainty': uncertainty_categories
    })

    # Create 1x4 plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot 1: Low-level by tumor status
    tumor_palette = {0: 'blue', 1: 'red'}
    sns.scatterplot(data=df, x='x_L', y='y_L', hue='tumor',
                   palette=tumor_palette, ax=axes[0])
    axes[0].set_title('Low-level by Tumor Status (Label)')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].legend(title='Tumor', labels=['Normal', 'Tumor'])

    # Plot 2: High-level by tumor status
    sns.scatterplot(data=df, x='x_H', y='y_H', hue='tumor',
                   palette=tumor_palette, ax=axes[1])
    axes[1].set_title('High-level by Tumor Status (Label)')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].legend(title='Tumor', labels=['Normal', 'Tumor'])

    # Plot 3: High-level by hospital
    hospital_palette = sns.color_palette('Set2', n_colors=5)
    sns.scatterplot(data=df, x='x_H', y='y_H', hue='hospital',
                   palette=hospital_palette, ax=axes[2])
    axes[2].set_title('High-level by Hospital (Environment)')
    axes[2].set_xlabel('t-SNE Dimension 1')
    axes[2].legend(title='Hospital', labels=[f'H{i}' for i in range(5)])

    # Plot 4: High-level by prediction uncertainty
    sns.scatterplot(data=df, x='x_H', y='y_H', hue='uncertainty',
                   palette='RdYlGn_r', ax=axes[3], hue_order=["Low", "Medium-Low", "Medium-High", "High"])
    axes[3].set_title('High-level by Model Uncertainty')
    axes[3].set_xlabel('t-SNE Dimension 1')
    axes[3].legend(title='Uncertainty', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def cvisualize_representations(model, test_loader, epoch):
    """Visualize Camelyon17 representations"""
    model.eval()
    z_L_all, z_H_all, labels, hospitals = [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y, e in test_loader:
            x = x.to(device)
            z_L, z_H, _ = model(x)
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y)
            hospitals.append(e.argmax(1))

    z_L_all = torch.cat(z_L_all).numpy()
    z_H_all = torch.cat(z_H_all).numpy()
    labels = torch.cat(labels).numpy()
    hospitals = torch.cat(hospitals).numpy()

    tsne = TSNE(n_components=2)
    z_L_2d = tsne.fit_transform(z_L_all)
    z_H_2d = tsne.fit_transform(z_H_all)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['blue', 'red']
    hospital_colors = plt.cm.Set2(np.linspace(0, 1, 5))

    # Plot by tumor status
    for ax, data, title in zip(axes[0], [z_L_2d, z_H_2d], ['Low-level', 'High-level']):
        for i, label in enumerate([0, 1]):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1],
                      c=colors[i], label=f'Tumor={label}')
        ax.set_title(f'{title} by Tumor Status')
        ax.legend()

    # Plot by hospital
    for ax, data, title in zip(axes[1], [z_L_2d, z_H_2d], ['Low-level', 'High-level']):
        for i in range(5):
            mask = hospitals == i
            if mask.any():
                ax.scatter(data[mask, 0], data[mask, 1],
                          c=[hospital_colors[i]], label=f'Hospital {i}')
        ax.set_title(f'{title} by Hospital')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'results/camelyon_representations_epoch_{epoch}.png')
    plt.close()
