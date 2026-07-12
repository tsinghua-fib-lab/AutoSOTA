import pandas as pd
import os


def get_baseline_results():
    results = {
        'PhysioNet': {
            'RNN': 2.92,
            'RNN-VAE': 5.93,
            'ODE-RNN': 2.23,
            'GRU-D': 3.33,
            'Latent ODE': 8.34,
            'LS4': 0.63,
            'iHT': 3.35
        },
        'USHCN': {
            'RNN': 4.32,
            'RNN-VAE': 7.56,
            'ODE-RNN': 2.47,
            'GRU-D': 3.40,
            'Latent ODE': 6.86,
            'LS4': 0.06,
            'iHT': 1.46
        }
    }
    return results


def create_comparison_table(dualfield_results):
    baseline = get_baseline_results()
    
    tables = {}
    
    for dataset in ['PhysioNet', 'USHCN']:
        data = []
        
        for method, mse in baseline[dataset].items():
            data.append({
                'Method': method,
                'MSE (×10^-3)': mse,
                'Type': 'Baseline'
            })
        
        if dataset in dualfield_results:
            for model_name, mse in dualfield_results[dataset].items():
                data.append({
                    'Method': model_name,
                    'MSE (×10^-3)': mse,
                    'Type': 'DualTimesField'
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('MSE (×10^-3)')
        tables[dataset] = df
    
    return tables


def print_results_table(tables):
    print("\n" + "="*100)
    print("Interpolation Results on Irregularly Sampled Time Series")
    print("="*100)
    
    for dataset, df in tables.items():
        print(f"\n{dataset} Dataset:")
        print("-"*100)
        print(df.to_string(index=False))
    
    print("\n" + "="*100)


def save_comparison_tables(tables, output_dir='./outputs/interpolation'):
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset, df in tables.items():
        output_file = os.path.join(output_dir, f'{dataset.lower()}_comparison.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved {dataset} comparison table to {output_file}")


def main():
    dualfield_results = {
        'PhysioNet': {
            'DualTimesField': None
        },
        'USHCN': {
            'DualTimesField': None
        }
    }
    
    output_dir = './outputs/interpolation'
    
    for dataset in ['PhysioNet', 'USHCN']:
        result_file = os.path.join(output_dir, f'{dataset.lower()}_results.csv')
        
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            mse_scaled = df['mse_scaled'].values[0]
            dualfield_results[dataset]['DualTimesField'] = mse_scaled
    
    tables = create_comparison_table(dualfield_results)
    print_results_table(tables)
    save_comparison_tables(tables, output_dir)


if __name__ == '__main__':
    main()
