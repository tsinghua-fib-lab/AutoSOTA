import subprocess
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SYNTHETIC
commands_synthetic = ['python3', '-m', 'experiments.main',
                      '--guarantee', 'p',
                      '--n_splits', '1',
                      '--lookahead', '0', '1',
                      '--lambda_range', '0.01', '0.5',
                      '--lambda_n', '200',
                      '--data_split', '0.5', '0.5', '0.0',
                      '--constraint_values', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4',
                      '--n_mc', '1000',
                      '--n_samples', '2000',
                      '--n_bins', '200',
                      '--learn_weights', 'true',
                      '--exp_folder_name', 'synthetic_p',
                      '--seed', '10603']

subprocess.run(commands_synthetic + ['--dataset', 'synthetic', '--name', 'syn_true1', '--gamma', '1.0'], cwd=parent_dir)
subprocess.run(commands_synthetic + ['--dataset', 'synthetic', '--name', 'syn_true2', '--gamma', '2.0'], cwd=parent_dir)
subprocess.run(commands_synthetic + ['--dataset', 'synthetic_unmeasured_confounder', '--name', 'syn_gamma1', '--gamma', '1.0'], cwd=parent_dir)
subprocess.run(commands_synthetic + ['--dataset', 'synthetic_unmeasured_confounder', '--name', 'syn_gamma2', '--gamma', '2.0'], cwd=parent_dir)
subprocess.run(commands_synthetic + ['--n_samples', '20000', '--data_split', '0.05', '0.05', '0.9', '--n_mc', '1000', '--dataset', 'synthetic_rct', '--name', 'syn_rct_gamma1', '--gamma', '1.0'], cwd=parent_dir)
subprocess.run(commands_synthetic + ['--n_samples', '20000', '--data_split', '0.05', '0.05', '0.9', '--n_mc', '1000', '--dataset', 'synthetic_rct', '--name', 'syn_rct_gamma2', '--gamma', '2.0'], cwd=parent_dir)

# STAR
commands_star = ['python3', '-m', 'experiments.main',
                 '--guarantee', 'p',
                 '--dataset', 'star',
                 '--n_splits', '3',
                 '--lookahead', '0', '1',
                 '--lambda_range', '0.01', '0.8',
                 '--lambda_n', '100',
                 '--data_split', '0.5', '0.25', '0.25',
                 '--constraint_values', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                 '--n_mc', '100',
                 '--n_bins', '40',
                 '--exp_folder_name', 'star_p',
                 '--seed', '10603']

subprocess.run(commands_star + ['--name', 'star', '--gamma', '1'], cwd=parent_dir)

# STROKE
commands_stroke = ['python3', '-m', 'experiments.main',
                   '--guarantee', 'p',
                   '--dataset', 'stroke',
                   '--n_splits', '3',
                   '--lookahead', '0', '1',
                   '--lambda_range', '0.01', '0.8',
                   '--lambda_n', '100',
                   '--data_split', '0.5', '0.25', '0.25',
                   '--constraint_values', '0.1', '0.15', '0.2', '0.225', '0.25', '0.3', '0.35', '0.4',
                   '--n_mc', '100',
                   '--n_bins', '40',
                   '--exp_folder_name', 'stroke_p',
                   '--seed', '10603']

subprocess.run(commands_stroke + ['--name', 'stroke', '--gamma', '1'], cwd=parent_dir)
