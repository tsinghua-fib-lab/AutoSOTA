config_path = 'configs/'

config = 'Tuebingen.gin'
for pair_id in range(1, 109):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'LSs.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'LS.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'MNU.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'AN.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'ANs.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'SIM.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'SIMc.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'SIMln.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'SIMG.gin'
for pair_id in range(1, 100+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'Cha.gin'
for pair_id in range(1, 300+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'Multi.gin'
for pair_id in range(1, 300+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)

config = 'Net.gin'
for pair_id in range(1, 300+1):
    command = f'python run_individual.py --config {config_path}{config} --pair_id {pair_id}'
    print(command)
