from evaluation import *
from utils import load_config

if __name__ == '__main__':
    cfg_path = "config.yaml"
    np.random.seed(42) 
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)
    I = int(rg_cfg["I"]) 
    J = int(rg_cfg["J"])
    M = int(rg_cfg["M"])
    c = list(rg_cfg["c"])
    data_size = int(rg_cfg["train_n"])
    test_size = int(rg_cfg["test_n"])
    epochs = int(rg_cfg["epochs"])
    b = np.zeros(J)
    A = np.zeros((J,I))
    
    # find_optimal_M(A, b, c=c, M_list=[100] + [i*i for i in range(20, 95, 5)], n_items=I, n_machines=J, save_path='./results/m.json')

    # evaluate_M_T_performance(A, b, c=c, M_list=[1000,2000,3000,4000,5000], n_items=I, n_machines=J, save_path='./results/ef_time.json')

    results = run_experiment(A, b, c=c, M=M, n_items=I, n_machines=J, data_size=data_size, K=epochs, data_path="./120.pkl")

