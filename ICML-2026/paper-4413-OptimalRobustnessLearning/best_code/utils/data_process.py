import pandas as pd
import os
import csv

if __name__ == '__main__':
    name_dict = {
        'LRU': 'LRU',

        'OracleLogDis': {
            'Predict[Belady]': 'ftp',
            'PredMark[Belady]': 'predmarker',
            'LMarker[Belady]': 'lmarker',
            'LNonMarker[Belady]': 'lnonmarker',
            'FollowerRobust[OracleState]': "fr",
            'Guard[Belady]-f-pred-no-relax': 'guardftp0',
            'Guard[Belady]-f-pred-relax-times-5': 'guardftp5',
            'CombDet[Predict[Belady], Marker]': 'blindoracledftp',
            'CombineRandom[Predict[Belady], Marker]': 'blindoraclerftp',
        },

        'OracleDis': {
            'Predict[Belady]': 'ftp',
            'PredMark[Belady]': 'predmarker',
            'LMarker[Belady]': 'lmarker',
            'LNonMarker[Belady]': 'lnonmarker',
            'FollowerRobust[OracleState]': "fr",
            'Guard[Belady]-f-pred-no-relax': 'guardftp0',
            'Guard[Belady]-f-pred-relax-times-5': 'guardftp5',
            'CombDet[Predict[Belady], Marker]': 'blindoracledftp',
            'CombineRandom[Predict[Belady], Marker]': 'blindoraclerftp',
        },

        'OracleBin': {
            'Predict[FBP]': 'fbp',
            'Mark0[FBP]': 'mark0',
            'Mark&Predict[OraclePhase]': 'markpred',
            'CombDet[Predict[FBP], Marker]': 'blindoracledfbp',
            'CombineRandom[Predict[FBP], Marker]': 'blindoraclerfbp',
            'Guard[FBP]-f-pred-no-relax': 'guardfbp0',
            'Guard[FBP]-f-pred-relax-times-5': 'guardfbp5',
        },

        'Parrot': {
            'LRU': 'LRU',
            'Marker': 'Marker',
            'Predict[Parrot]': 'FTP',
            'PredMark[Parrot]': 'PredMark',
            'LMarker[Parrot]': 'LMarker',
            'LNonMarker[Parrot]': 'LNonMarker',
            'CombDet[Predict[Parrot], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[Parrot], Marker]': 'BlindOracleR',
            'FollowerRobust[ParrotState]': 'F&R',
            'Guard[Parrot]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[Parrot]-f-pred-relax-times-5': 'Guard&FTP5',
        },

        'PLECO': {
            'LRU': 'LRU',
            'Marker': 'Marker',
            'Predict[PLECO]': 'FTP',
            'PredMark[PLECO]': 'PredMark',
            'LMarker[PLECO]': 'LMarker',
            'LNonMarker[PLECO]': 'LNonMarker',
            'CombDet[Predict[PLECO], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[PLECO], Marker]': 'BlindOracleR',
            'FollowerRobust[PLECOState]': 'F&R',
            'Guard[PLECO]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[PLECO]-f-pred-relax-times-5': 'Guard&FTP5',
        },

        'POPU': {
            'LRU': 'LRU',
            'Marker': 'Marker',
            'Predict[POPU]': 'FTP',
            'PredMark[POPU]': 'PredMark',
            'LMarker[POPU]': 'LMarker',
            'LNonMarker[POPU]': 'LNonMarker',
            'CombDet[Predict[POPU], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[POPU], Marker]': 'BlindOracleR',
            'FollowerRobust[POPUState]': 'F&R',
            'Guard[POPU]-f-pred-no-relax': 'Guard&FTP0',
            'Guard[POPU]-f-pred-relax-times-5': 'Guard&FTP5',
        },

        'PLECOBin': {
            'LRU': 'LRU',
            'Marker': 'Marker',
            'Predict[PLECOBin]': 'FBP',
            'Mark0[PLECOBin]': 'Mark0',
            'CombDet[Predict[PLECOBin], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[PLECOBin], Marker]': 'BlindOracleR',
            'Guard[PLECOBin]-f-pred-no-relax': 'Guard&FBP0',
            'Guard[PLECOBin]-f-pred-relax-times-5': 'Guard&FBP5',
        },

        'GBMBin': {
            'LRU': 'LRU',
            'Marker': 'Marker',
            'Predict[GBMBin]': 'FBP',
            'Mark0[GBMBin]': 'Mark0',
            'CombDet[Predict[GBMBin], Marker]': 'BlindOracleD',
            'CombineRandom[Predict[GBMBin], Marker]': 'BlindOracleR',
            'Guard[GBMBin]-f-pred-no-relax': 'Guard&FBP0',
            'Guard[GBMBin]-f-pred-relax-times-5': 'Guard&FBP5',
        },

    }

    mode_enums = ['avg', 'lru_avg', 'bar', 'frac_bar', 'plot']

    mode = 'avg'
    predictor = 'Parrot'
    root_dir_path = 'dump/stat_parrot'
    skip_bk_citi = True
    
    print(mode)

    if mode == 'plot':
        skip_bk_citi = False
        res_csv_path = 'plot_res'
    
    res_dict = {}
    for dirpath in os.listdir(root_dir_path):
        dataset = dirpath
        if skip_bk_citi and (dataset == 'brightkite' or dataset == 'citi'):
            continue

        full_path = os.path.join(root_dir_path, dirpath)
        if os.path.isdir(full_path):
            if mode == 'avg':
                res_dict[dataset] = {}
                if predictor == 'GBMBin':
                    this_path = os.path.join(full_path, '1', 'gbm.csv')
                elif predictor == 'PLECO' or predictor == 'POPU' or predictor == 'PLECOBin':
                    this_path = os.path.join(full_path, '1', 'pleco_popu_pleco-bin.csv')
                elif predictor == 'Parrot':
                    this_path = os.path.join(full_path, '1', 'parrot.csv')
                else:
                    raise ValueError(f'no pred {predictor}')
                
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    for name in result_dict:
                        cr = result_dict[name]['Competitive Ratio']
                        if name in name_dict[predictor]:
                            res_name = name_dict[predictor][name]
                            if res_name not in res_dict[dataset]:
                                res_dict[dataset][res_name] = []
                            res_dict[dataset][res_name].append(cr)
            elif mode == 'lru_avg':
                res_dict[dataset] = {}
                if predictor == 'GBMBin':
                    this_path = os.path.join(full_path, '1', 'gbm.csv')
                elif predictor == 'PLECO' or predictor == 'POPU' or predictor == 'PLECOBin':
                    this_path = os.path.join(full_path, '1', 'pleco_popu_pleco-bin.csv')
                elif predictor == 'Parrot':
                    this_path = os.path.join(full_path, '1', 'parrot.csv')
                else:
                    raise ValueError(f'no pred {predictor}')
                
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    lru = result_dict['LRU']['Competitive Ratio']
                    for name in result_dict:
                        cr = result_dict[name]['Competitive Ratio']
                        if name in name_dict[predictor]:
                            res_name = name_dict[predictor][name]
                            if res_name not in res_dict[dataset]:
                                res_dict[dataset][res_name] = []
                            res_dict[dataset][res_name].append((cr-1)/(lru-1))
            elif mode == 'frac_bar':
                res_dict[dataset] = {}
                for frac_path in os.listdir(full_path):
                    frac = float(frac_path)
                    if predictor == 'GBMBin':
                        this_path = os.path.join(full_path, frac_path, 'gbm.csv')
                    else:
                        raise ValueError(f'no pred {predictor}')
                    if os.path.exists(this_path):
                        df = pd.read_csv(this_path)
                        result_dict = df.set_index('Name').T.to_dict('dict')
                        lru = result_dict['LRU']['Competitive Ratio']
                        if name in name_dict[predictor]:
                            res_name = name_dict[predictor][name]
                            if res_name not in res_dict[dataset]:
                                res_dict[dataset][res_name] = []
                            res_dict[dataset][res_name].append((frac, (cr-1)/(lru-1)))
            elif mode == 'bar':         
                if predictor == 'GBMBin':
                    this_path = os.path.join(full_path, '1', 'gbm.csv')
                elif predictor == 'PLECO' or predictor == 'POPU' or predictor == 'PLECOBin':
                    this_path = os.path.join(full_path, '1', 'pleco_popu_pleco-bin.csv')
                elif predictor == 'Parrot':
                    this_path = os.path.join(full_path, '1', 'parrot.csv')
                else:
                    raise ValueError(f'no pred {predictor}')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    lru = result_dict['LRU']['Competitive Ratio']
                    for name in result_dict:
                        cr = result_dict[name]['Competitive Ratio']
                        if name in name_dict[predictor]:
                            res_name = name_dict[predictor][name]
                            if res_name not in res_dict:
                                res_dict[res_name] = []
                            res_dict[res_name].append((dataset, (cr-1)/(lru-1)))
            elif mode == 'plot':
                if predictor == 'OracleBin':
                    prefix = 'bin'
                    start = 'Bin-'
                elif predictor == 'OracleLogDis':
                    prefix = 'logdis'
                    start = 'LogDis-'
                elif predictor == 'OracleDis':
                    prefix = 'dis'
                    start = 'Dis-'

                this_dataset_plot_path = os.path.join(res_csv_path, dataset, prefix)
                os.makedirs(this_dataset_plot_path, exist_ok=True)
                dir_path = os.path.join(full_path, '1')                
                this_path = os.path.join(dir_path, f'{prefix}.csv')
                if os.path.exists(this_path):
                    df = pd.read_csv(this_path)
                    result_dict = df.set_index('Name').T.to_dict('dict')
                    for name in result_dict:
                        if name in name_dict[predictor]:
                            alg_name = name_dict[predictor][name]
                            res_path = os.path.join(this_dataset_plot_path, f'{alg_name}.csv')
                            tuple_list = []
                            for noise, value in result_dict[name].items():
                                if noise.startswith(start):
                                    this_x = noise.split('-')[1]
                                    this_y = value.split('/')[1]
                                    tuple_list.append((this_x, this_y))
                            with open(res_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(['x', 'y'])
                                for tup in tuple_list:
                                    writer.writerow(tup)
                            print(f"Data has been written to {res_path}")
    if mode == 'avg' or mode == 'lru_avg':
        # avg
        sum_d = {}
        cnt_d = {}
        for dataset, d in sorted(res_dict.items()):
            for name, l in d.items():
                if name not in sum_d:
                    sum_d[name] = 0
                    cnt_d[name] = 0
                
                sum_d[name] += l[0]
                cnt_d[name] += 1

        for name in sum_d.keys():
            print(f'{name}: {sum_d[name]/cnt_d[name]}')

    elif mode == 'frac_bar':
        for dataset, d in sorted(res_dict.items()):
            print(dataset)
            for name, l in d.items():
                print(name)
                sorted_list = sorted(l, key=lambda x: x[0])
                for t in sorted_list:
                    print(f'({t[0]},{t[1]})')
    elif mode == 'bar':
        for name, l in res_dict.items():
            sorted_list = sorted(l, key=lambda x: x[0])
            print(name)
            for t in sorted_list:
                print(f'({t[0]},{t[1]})')

            
