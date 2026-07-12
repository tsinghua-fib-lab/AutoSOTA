import numpy as np
import matplotlib.pyplot as plt

def pac_plot(error, percent_saved, epsilon, Y, Yhat, confidence, loss, plot_title, num_trials=1000, num_plotted=50, xlim=[0,0.4], plot_naive=True):
    
    selected_indices = np.random.choice(num_trials, size=num_plotted, replace=False)
    
    plt.figure(figsize=(6, 4))
    
    plt.scatter(
        error[selected_indices],
        percent_saved[selected_indices],
        marker='x',
        s=60,
        color='#2274A5',
        alpha=0.9,
        label='PAC labeling',
        linewidths=2
    )
    
    plt.axvline(x=epsilon, linestyle='--', color='black', linewidth=1.2, alpha=0.6)
    plt.xlim(xlim)
    plt.ylim(-1, 105)
    plt.yticks(fontsize=12)
    
    xticks = plt.xticks()[0]
    xticks = np.append(xticks, epsilon)
    xticks = np.unique(np.round(xticks, 3))
    plt.xticks(xticks)
    
    plt.gca().set_xticklabels(
        [r'$\varepsilon=$' + str(epsilon) if np.isclose(tick, epsilon) else f'{tick:.2f}' for tick in xticks],
        fontsize=12
    )
    
    
    thresh = epsilon
    Y_til = Yhat.copy()
    Y_til[confidence < 1-thresh] = Y[confidence < 1 - thresh]
    percent_saved = np.mean(confidence >= 1 - thresh)*100

    if plot_naive:
        plt.scatter(
            loss(Y, Y_til),
            percent_saved,
            marker='x',
            s=70,
            facecolor='#F7B05B', 
            alpha=0.9,
            label='naive threshold',
            linewidths=2
        )
    
    plt.scatter(
        loss(Y, Yhat),
        100,
        marker='x',
        s=70,
        facecolor='#F75C03',
        alpha=0.9,
        label='AI only',
        linewidths=2
    )
    
    # Axis labels and limits
    plt.xlabel('error', fontsize=16)
    plt.ylabel('budget save (%)', fontsize=16)
    
    
    # Add grid with subtle style
    plt.grid(True, linestyle=':', color='gray', alpha=0.4)
    
    # Add legend with smart placement
    plt.legend(frameon=False, fontsize=16, loc='lower right')
    
    # Remove top and right borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Clean layout
    plt.tight_layout()
    plt.savefig(plot_title + str(epsilon) + ".pdf")
    plt.show()
        

def pac_router_plot(errs, budget_saves, epsilon, plot_title, num_trials=1000, num_plotted=50, xlim=[0,0.3], ylim=None, cost_free=True):
    
    selected_indices = np.random.choice(num_trials, size=num_plotted, replace=False)
    plt.figure(figsize=(6, 4))
    
    
    plt.figure(figsize=(6, 5))
    
    plt.scatter(
        errs[0][selected_indices],
        budget_saves[0][selected_indices],
        marker='x',
        s=60,
        color='#2274A5',
        alpha=0.9,
        label='router',
        linewidths=2
    )
    
    plt.scatter(
        errs[1][selected_indices],
        budget_saves[1][selected_indices],
        marker='x',
        s=60,
        color='#00CC66',
        alpha=0.9,
        label='GPT',
        linewidths=2
    )
    
    plt.scatter(
        errs[2][selected_indices],
        budget_saves[2][selected_indices],
        marker='x',
        s=60,
        color='#F75C03',
        alpha=0.9,
        label='Claude',
        linewidths=2
    )
    
    plt.axvline(x=epsilon, linestyle='--', color='black', linewidth=1.2, alpha=0.6)

    if ylim is None:
        if cost_free:
            plt.ylim(-1, 105)
    else:
        plt.ylim(ylim)
    plt.xlim(xlim)
    xticks = plt.xticks()[0]
    xticks = np.append(xticks, epsilon)
    xticks = np.unique(np.round(xticks, 3))
    plt.xticks(xticks)
    plt.gca().set_xticklabels(
        [r'$\varepsilon=$' + str(epsilon) if np.isclose(tick, epsilon) else f'{tick:.2f}' for tick in xticks],
        fontsize=12
    )
    plt.yticks(fontsize=12)
    
    plt.xlabel('error', fontsize=16)
    if cost_free:
        plt.ylabel('budget save (%)', fontsize=16)
    else:
        plt.ylabel('save in cost', fontsize=16)
    
    plt.grid(True, linestyle=':', color='gray', alpha=0.4)
    
    plt.legend(frameon=False, fontsize=16, loc='lower right')
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plot_title + str(epsilon) + ".pdf")
    plt.show()