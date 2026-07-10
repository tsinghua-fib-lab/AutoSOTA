from matplotlib import pyplot as plt

# Visualization helper
def plot_ssl_pairs(v1, v2):
    fig, axes = plt.subplots(2, 10, figsize=(10 * 2, 4))
    
    for i in range(10):
        # Top row: View 1
        axes[0, i].imshow(v1[i][0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("View 1 (Anchor)")
        
        # Bottom row: View 2
        axes[1, i].imshow(v2[i][0].cpu(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("View 2 (Positive)")
            
    plt.suptitle("Self-Supervised Learning Pairs (Positive Pairs)")
    plt.tight_layout()
    plt.show()