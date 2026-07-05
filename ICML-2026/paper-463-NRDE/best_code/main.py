from NRDE import NRDE_run
from NRDE import read_data,read_gauss_noise_data,read_contaminated_data

file_path = "datasets/35_SpamBase.npz"
train_data, train_labels, test_data, test_labels, _, _ = read_data(file_path, normalization='z-score')
    #np.random.seed(42)
    #train_data = np.random.randn(1000, 10)
    #train_labels = np.zeros(1000)
    #test_data = np.random.randn(500, 10)
    #test_labels = np.random.choice([0, 1], size=500, p=[0.95, 0.05])
    
auroc, auprc = NRDE_run(
        train_data, train_labels, test_data, test_labels,
        lr=0.001, grad_pun=0.1, n_epochs=10, bs=512, mid_dim=2048,
        verbose=True
    )
print(f"\nFinal result: AUROC = {auroc:.3f}, AUPRC = {auprc:.3f}")
