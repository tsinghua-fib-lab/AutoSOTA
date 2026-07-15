import concurrent.futures
from multiprocessing import cpu_count

import math
import logging
import preprocessing
import ourAlg
import baseline
import show
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
repeat_times = 2

def calc_mu_std(sum_err, sum_err2, count):
    mu = sum_err / count
    std = math.sqrt((sum_err2 - sum_err * mu) / (count - 1))
    return (mu, std)

def implement_epsilon_test(datasetName, n, d, max_workers_list):
    epsilon_test = [x / 10.0 for x in range(5, 45, 5)]
    print('epsilon_test: ', epsilon_test)

    patterns = ['triangle', 'edge', '2star']
    for i in range(0, 3):
        pattern = patterns[i]
        max_workers = max_workers_list[i]

        logger.info(f"Starting epsilon test of {datasetName} for pattern: {pattern}")
        (d_max, test_input_nodes, edges, h, m) = preprocessing.graph_data_load(datasetName, pattern, n, logger, d)
        Q_num = math.ceil(n**(1.5)) # d = 1
        # Q_num = n * n # for d = 2
        # Q = preprocessing.generate_skewed_queries(Q_num, m, d) # d = 1
        Q = preprocessing.generate_queries(Q_num, m, d) # d = 1
        # Q = preprocessing.generate_random_queries(Q_num, m, d) # d = 2
        logger.info(f"Query generation finished!")
        
        # true = ourAlg.query_true(n, m, d, Q, test_input_nodes, logger)
        # err_ourAlg_pure = ourAlg.pure_DP(n, m, d, Q, 2.0, test_input_nodes, pattern, logger)
        # err_ourAlg_approx = ourAlg.approx_DP(n, m, d, Q, d_max, 2.0, 0.00001, test_input_nodes, pattern, logger)
        # err_base_comp_ADP = baseline.base_comp_ADP(n, Q_num, 1, d_max, 2.0, 0.00001, true, pattern, logger) 
        # print(err_ourAlg_pure, err_ourAlg_approx)
        # logger.info('single pass for DP finished!')
        # exit(0)
        
        result_file_name = f'{datasetName}/{datasetName}_{pattern}_epsilon_result.csv'
        delta = 0.00001
        with open(result_file_name, mode='w', newline='') as csv_file:
            fieldnames = ['epsilon', 'pure_DP_error', 'pure_DP_std', 'approx_DP_error', 'approx_DP_std', 
                          'base_comp_error', 'base_comp_std', 'base_comp_ADP_error', 'base_comp_ADP_std']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_true = executor.submit(ourAlg.query_true, n, m, d, Q, test_input_nodes, logger)
                future_pure = [None] * repeat_times
                future_approx = [None] * repeat_times
                for t in range(repeat_times):
                    future_pure[t] = [executor.submit(ourAlg.pure_DP, n, m, d, Q, eps, test_input_nodes, pattern, logger) for eps in epsilon_test]
                    future_approx[t] = [executor.submit(ourAlg.approx_DP, n, m, d, Q, d_max, eps, delta, test_input_nodes, pattern, logger) for eps in epsilon_test]
                
                true = future_true.result()
                err_ourAlg_pure = []
                err_ourAlg_approx = []
                count = Q_num * repeat_times
                for i, eps in enumerate(epsilon_test):
                    pure_sum_err = 0.0
                    pure_sum_err2 = 0.0
                    approx_sum_err = 0.0
                    approx_sum_err2 = 0.0
                    for t in range(repeat_times):
                        err, err2 = future_pure[t][i].result()
                        pure_sum_err += err
                        pure_sum_err2 += err2
                        err, err2 = future_approx[t][i].result()
                        approx_sum_err += err
                        approx_sum_err2 += err2
                    err_ourAlg_pure.append(calc_mu_std(pure_sum_err, pure_sum_err2, count))
                    err_ourAlg_approx.append(calc_mu_std(approx_sum_err, approx_sum_err2, count))
                    
                for i, eps in enumerate(epsilon_test):
                    err_base_comp = baseline.base_comp(n, Q_num, repeat_times, eps, true, pattern, logger)
                    err_base_comp_ADP = baseline.base_comp_ADP(n, Q_num, repeat_times, d_max, eps, delta, true, pattern, logger) 
                    writer.writerow({
                        'epsilon': eps, 
                        'pure_DP_error': err_ourAlg_pure[i][0], 
                        'pure_DP_std': err_ourAlg_pure[i][1],
                        'approx_DP_error': err_ourAlg_approx[i][0],
                        'approx_DP_std': err_ourAlg_approx[i][1],
                        'base_comp_error': err_base_comp[0],
                        'base_comp_std': err_base_comp[1],
                        'base_comp_ADP_error': err_base_comp_ADP[0],
                        'base_comp_ADP_std': err_base_comp_ADP[1]
                    })
        picture_file_name = f'{datasetName}/{datasetName}_{pattern}_epsilon_result.png'
        show.epsilon_show(result_file_name, picture_file_name, pattern.capitalize())
        

def implement_Q_test(datasetName, n, m, d, max_workers_list, lim_Q):
    Q_num_test = [math.ceil(n**(1 + i/10.0)) for i in range(6)]
    Q_num_test.append(math.ceil(n**(1.55)))
    Q_num_test.append(math.ceil(n**(1.6))) 
    print('Q_num_test: ', Q_num_test)
    Q_test = [None] * 8
    
    for i, Q_num in enumerate(Q_num_test):
        # Q_test[i] = preprocessing.generate_skewed_queries(Q_num, m, d) # d = 1
        Q_test[i] = preprocessing.generate_queries(Q_num, m, d) # d = 1
    logger.info(f"Query generation finished!")
    
    patterns = ['triangle', 'edge', '2star']
    for i in range(0, 3):
        pattern = patterns[i]
        max_workers = max_workers_list[i]
        
        logger.info(f"Starting Q test of {datasetName} for pattern: {pattern}")
        (d_max, test_input_nodes, edges, h, m) = preprocessing.graph_data_load(datasetName, pattern, n, logger, d)
    
        # true = ourAlg.query_true(n, d, Q, test_input_nodes, logger)
        # err_ourAlg_pure = ourAlg.pure_DP(n, d, Q, 2.0, test_input_nodes, pattern, logger)
        # err_ourAlg_approx = ourAlg.approx_DP(n, d, Q, d_max, 2.0, 0.00001, test_input_nodes, pattern, logger)
        # err_base_comp_ADP = baseline.base_comp_ADP(n, Q_num, d_max, 2.0, 0.00001, true, pattern, logger) 
        # print(max(true))
        # print(err_ourAlg_pure, err_ourAlg_approx, err_base_comp_ADP)
        # logger.info('single pass for DP finished!')
        # exit(0)
        
        result_file_name = f'{datasetName}/{datasetName}_{pattern}_Q_result.csv'
        eps = 2.0
        delta = 0.00001
        with open(result_file_name, mode='w', newline='') as csv_file:
            fieldnames = ['Q_num', 'pure_DP_error', 'pure_DP_std', 'approx_DP_error', 'approx_DP_std', 
                          'base_comp_error', 'base_comp_std', 'base_comp_ADP_error', 'base_comp_ADP_std']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_true = [executor.submit(ourAlg.query_true, n, m, d, Q, test_input_nodes, logger) for Q in Q_test]
                future_pure = [None] * repeat_times
                future_approx = [None] * repeat_times
                for t in range(repeat_times):
                    future_pure[t] = [executor.submit(ourAlg.pure_DP, n, m, d, Q, eps, test_input_nodes, pattern, logger) for Q in Q_test]
                    future_approx[t] = [executor.submit(ourAlg.approx_DP, n, m, d, Q, d_max, eps, delta, test_input_nodes, pattern, logger) for Q in Q_test]
                
                true = [future_true[i].result() for i, Q_num in enumerate(Q_num_test)]
                err_ourAlg_pure = []
                err_ourAlg_approx = []
                for i, Q_num in enumerate(Q_num_test):
                    pure_sum_err = 0.0
                    pure_sum_err2 = 0.0
                    approx_sum_err = 0.0
                    approx_sum_err2 = 0.0
                    for t in range(repeat_times):
                        err, err2 = future_pure[t][i].result()
                        pure_sum_err += err
                        pure_sum_err2 += err2
                        err, err2 = future_approx[t][i].result()
                        approx_sum_err += err
                        approx_sum_err2 += err2
                    count = Q_num * repeat_times
                    err_ourAlg_pure.append(calc_mu_std(pure_sum_err, pure_sum_err2, count))
                    err_ourAlg_approx.append(calc_mu_std(approx_sum_err, approx_sum_err2, count))
                    
                for i, Q_num in enumerate(Q_num_test):
                    err_base_comp = baseline.base_comp(n, Q_num, repeat_times, eps, true[i], pattern, logger)
                    err_base_comp_ADP = baseline.base_comp_ADP(n, Q_num, repeat_times, d_max, eps, delta, true[i], pattern, logger) 
                    writer.writerow({
                        'Q_num': Q_num, 
                        'pure_DP_error': err_ourAlg_pure[i][0], 
                        'pure_DP_std': err_ourAlg_pure[i][1],
                        'approx_DP_error': err_ourAlg_approx[i][0],
                        'approx_DP_std': err_ourAlg_approx[i][1],
                        'base_comp_error': err_base_comp[0],
                        'base_comp_std': err_base_comp[1],
                        'base_comp_ADP_error': err_base_comp_ADP[0],
                        'base_comp_ADP_std': err_base_comp_ADP[1]
                    })
        picture_file_name = f'{datasetName}/{datasetName}_{pattern}_Q_result.png'
        show.Q_show(result_file_name, picture_file_name, pattern.capitalize(), lim_Q)
        
def query_time_test(datasetName, n, m, d):
    eps = 2.0
    delta = 0.00001
    Q_num = int(math.ceil(n**1.6))
    # Q = preprocessing.generate_skewed_queries(Q_num, m, d) # d = 1
    Q = preprocessing.generate_queries(Q_num, m, d) # d = 1
    # Q = preprocessing.generate_random_queries(Q_num, m, d) # d = 2
    
    patterns = ['triangle', 'edge', '2star']
    for i in range(0, 3):
        pattern = patterns[i]
        (d_max, test_input_nodes, edges, h, m) = preprocessing.graph_data_load(datasetName, pattern, n, logger, d)
        
        # Unit: microseconds
        base_comp_qtime, base_comp_se = baseline.base_comp_qtime(n, d, Q[:repeat_times], eps, edges, h, pattern)
        logger.info('finish calculating query time for basic compoisition')
        base_comp_ADP_qtime, base_comp_ADP_se = baseline.base_comp_ADP_qtime(n, d, Q[:repeat_times], d_max, eps, delta, edges, h, pattern)
        logger.info('finish calculating query time for advanced compoisition')
        approx_DP_qtime, approx_DP_se = ourAlg.approx_DP_qtime(n, m, d, Q, d_max, eps, delta, test_input_nodes, pattern)
        logger.info('finish calculating query time for approximate DP')
        pure_DP_qtime, pure_DP_se = ourAlg.pure_DP_qtime(n, m, d, Q, eps, test_input_nodes, pattern)
        logger.info('finish calculating query time for pure DP')
        # print(pure_DP_se / pure_DP_qtime, approx_DP_se / approx_DP_qtime, base_comp_se / base_comp_qtime, base_comp_ADP_se / base_comp_ADP_qtime)
        
        result_file_name = f'{datasetName}/{datasetName}_{pattern}_qtime_result.csv'
        
        with open(result_file_name, mode='w', newline='') as csv_file:
            fieldnames = ['pure_DP', 'approx_DP', 'base_comp', 'base_comp_ADP']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'pure_DP': pure_DP_qtime,
                'approx_DP': approx_DP_qtime,
                'base_comp': base_comp_qtime,
                'base_comp_ADP': base_comp_ADP_qtime
            })

def preprocessing_time_test(datasetName, n, d):
    eps = 2.0
    delta = 0.00001
    
    patterns = ['triangle', 'edge', '2star']
    for i in range(0, 3):
        pattern = patterns[i]
        (d_max, test_input_nodes, edges, h, m) = preprocessing.graph_data_load(datasetName, pattern, n, logger, d)
        
        # Unit: minutes 
        base_comp_ADP_prtime, base_comp_ADP_se = baseline.base_comp_ADP_prtime(n, d, edges, h, repeat_times, pattern)
        logger.info('finish calculating preprocessing time for advanced compoisition')
        approx_DP_prtime, approx_DP_se = ourAlg.approx_DP_prtime(n, m, d, eps, delta, edges, h, repeat_times, pattern)
        logger.info('finish calculating preprocessing time for approximate DP')
        pure_DP_prtime, pure_DP_se = ourAlg.pure_DP_prtime(n, m, d, eps, edges, h, repeat_times, pattern)
        logger.info('finish calculating preprocessing time for pure DP')
        # print(pure_DP_se / base_comp_ADP_prtime, approx_DP_se / base_comp_ADP_prtime, base_comp_ADP_se / base_comp_ADP_prtime)
        
        result_file_name = f'{datasetName}/{datasetName}_{pattern}_prtime_result.csv'
        with open(result_file_name, mode='w', newline='') as csv_file:
            fieldnames = ['pure_DP', 'approx_DP', 'base_comp', 'base_comp_ADP']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'pure_DP': pure_DP_prtime,
                'approx_DP': approx_DP_prtime,
                'base_comp': 0,
                'base_comp_ADP': base_comp_ADP_prtime
            })
        
if __name__ == "__main__":
    # d = 2
    # dataset_names = ["ca-netscience"]
    # n_list = [379]
    # m_list = [379]
    # max_workers_lists = [[40, 40, 40]]
    # lim_Q = [5]
    
    d = 1
    dataset_names = ["musae-squirrel", "bio-WormNet-v3"]
    n_list = [5201, 16347]
    m_list = [5201, 16347] 
    max_workers_lists = [[120, 120, 120], [75, 62, 16]]
    lim_Q = [5, 6]
    
    for i in range(0, 2):
        preprocessing_time_test(dataset_names[i], n_list[i], d)
        query_time_test(dataset_names[i], n_list[i], m_list[i], d)
        implement_epsilon_test(dataset_names[i], n_list[i], d, max_workers_lists[i])
        implement_Q_test(dataset_names[i], n_list[i], m_list[i], d, max_workers_lists[i], lim_Q[i])
