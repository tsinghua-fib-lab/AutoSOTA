import pandas as pd


def dataToCsv(file, data, rows, columns, target):
    data = list(data)
    columns = list(columns)
    rows = list(rows)
    file_data = pd.DataFrame(data, index=rows, columns=columns)
    file_target = pd.DataFrame(target, index=rows, columns=['Equilibrium Parameter'])
    file_all = file_data.join(file_target, how='outer')
    file_all.to_csv(file)
