import pickle
import numpy as np
import gurobipy as gp
from apub import APUB
import time
from saa import SAA


def evaluate_dimension(A, b, c, n_items, n_machines):
    """Worker function to process a single M value"""
    tt1 = []
    tt2 = []
    tt3 = []
    cuts = []
    saa_iter = []
    xi_samples_list = []
    np.random.seed(1234)
    random_numbers = np.random.randint(0, 200, size=30)

    with open(f"./samples/samples2.pkl", "rb") as f:
        xi_samples_list = pickle.load(f)['train_samples']
    
    for i in random_numbers:
        xi_samples_list[i]['T'] = np.vstack((xi_samples_list[i]['T'][:n_machines, :n_items],np.zeros(n_items)))
        temp = np.zeros((120, n_machines+1))
        temp[:,-1] = xi_samples_list[i]['h'][:,-1]
        xi_samples_list[i]['h'] = temp
        apub = APUB(A, b, c=c, n_items=n_items, n_machines=n_machines, model=gp.Model())
        start1 = time.perf_counter()
        apub.extensive_form(xi_samples_list[i], alpha=0.1, M_bootstrap=5000)
        end1 = time.perf_counter()
        
        # L-shape method timing
        start2 = time.perf_counter()
        _, _, _, num_optimal_cuts = apub.solve_two_stage_apub(
            xi_samples_list[i],
            alpha=0.1,
            M_bootstrap=5000
        )
        end2 = time.perf_counter()

        saa = SAA(model=gp.Model(), c=c, n_items=n_items, n_machines=n_machines)
        start3 = time.perf_counter()
        _,_,it=saa.solve_nf(xi_samples_list[i], max_iter=30, tol=1e-4)
        end3 = time.perf_counter()
        
        tt1.append(end1 - start1)
        tt2.append(end2 - start2)
        tt3.append(end3 - start3)
        cuts.append(num_optimal_cuts)
        saa_iter.append(it)
    
    return {
        'extensive_form_time': tt1,
        'lshape_time': tt2,
        'saa_time': tt3,
        'cuts': cuts,
        'saa_iterations': saa_iter
    }


if __name__ == "__main__":
    # Example usage
    for I,J in [(10,4),(40,16),(80,32)]:
        print(f"Evaluating for I={I}, J={J}")
        c = [-14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5,
            -14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5,
            -14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5,
            -14, -9, -20, -15, -4, -40, -18, -11, -13, -16, -17, -8, -9, -24, -10, -7, -12, -3, -4, -5]
        b = np.zeros(J)
        A = np.zeros((J,I))

        results = evaluate_dimension(A, b, c[:I], I, J)
        print(f"Results for I={I}, J={J}: ef_time={np.mean(results['extensive_form_time']),np.std(results['extensive_form_time'])},\
              lshape_time={np.mean(results['lshape_time']),np.std(results['lshape_time'])}, cuts={np.mean(results['cuts']),np.std(results['cuts'])}, \
              saa_iterations={np.mean(results['saa_iterations']),np.std(results['saa_iterations'])}, saa_time={np.mean(results['saa_time']),np.std(results['saa_time'])}")