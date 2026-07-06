import os


def generate_benchmark_table():
    res_root = ''
    data_lst = ['CHAMELEON', 'CAMO', 'COD10K']
    model_lst = ['']

    for i in range(len(model_lst)):
        for j in range(len(data_lst)):
            txt_path = os.path.join(res_root, '{}_result.txt'.format(data_lst[j]))
            if not os.path.exists(txt_path):
                print('& -   & -   & -   & -', end='\n')
            else:
                with open(txt_path) as f:
                    line_ori = f.readlines()
                    for k in range(len(line_ori)):
                        line = line_ori[k]
                        if line.split('Model:')[1].split(') Smeasure')[0] in model_lst[i]:
                            if 'NaN' in line:
                                S_measure = '-'
                                w_F = '-'
                                mean_E_m = '-'
                                MAE = '-'
                            else:
                                S_measure = line.split('Smeasure:')[1].split('; wFmeasure')[0]
                                w_F = line.split('wFmeasure:')[1].split(';MAE')[0]
                                mean_E_m = line.split('meanEm:')[1].split('; maxEm')[0]
                                MAE = line.split('MAE:')[1].split('; adpEm')[0]

                            print('& {}   & {}   & {}   & {}'.format(S_measure, mean_E_m, w_F, MAE), end='\n')


if __name__ == '__main__':
    generate_benchmark_table()