import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    df = pd.DataFrame()
    for alpha_train in ["0.20", "0.25", "0.33", "0.50", "1.00", "2.00", "3.00", "4.00", "5.00"]:
        log_file_path = f'/home/sheng136/DeconDTN/code/output/logs/global-a0/bert-base-uncased_atrain-{alpha_train}.log'
        df = pd.concat([df, get_data(log_file_path, alpha_train)], ignore_index=True)

    df.to_csv('/home/sheng136/DeconDTN/code/misc/csv/mask_sizes.csv', index=False)


# Initialize an empty list to store extracted data
def get_data(log_file_path, alpha_train):
    # Initialize an empty list to store extracted data

    data = []

    # Define regular expression patterns to extract relevant information
    pattern_masking = re.compile(r'Masking (intersection|compliment|all) of top ([\d.]+)% changed weights from both models')
    pattern_size = re.compile(r'size of the (\w+) mask is (\d+), which is ([\d.]+)% of the trainable parameters in (\w+) model')

    # Initialize variables to hold the current type and percentage
    current_mask_type = None
    current_top_x_percent = None

    # Read and process the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            masking_match = pattern_masking.search(line)
            size_match = pattern_size.search(line)
            
            if masking_match:
                current_mask_type = masking_match.group(1)
                current_top_x_percent = float(masking_match.group(2))
            
            if size_match and current_mask_type is not None and current_top_x_percent is not None:
                mask_type = size_match.group(1)
                size = int(size_match.group(2))
                size_percentage = float(size_match.group(3))
                model = size_match.group(4)
                data.append({
                    'mask_type': current_mask_type,
                    'size': size,
                    'size_percentage': size_percentage,
                    'model': model,
                    'top_x_percent': current_top_x_percent,
                })
                current_mask_type = None
                current_top_x_percent = None  # Reset for next entry

    # Convert the extracted data to a DataFrame
    df = pd.DataFrame(data)
    # add 0 entries
    intact_row  = [
        {'mask_type': 'all', 'size': 0, 'size_percentage': 0, 'model': 'dementia', 'top_x_percent': 0},
        {'mask_type': 'compliment', 'size': 0, 'size_percentage': 0, 'model': 'dementia', 'top_x_percent': 0},
        {'mask_type': 'intersection', 'size': 0, 'size_percentage': 0, 'model': 'dementia', 'top_x_percent': 0}
    ]
    new_row_df = pd.DataFrame(intact_row)
    df = pd.concat([new_row_df, df], ignore_index=True)


    df = df.assign(alpha_train=alpha_train)
    
    return df


if __name__ == '__main__':
    main()
# Summarize the data
# summary = df.groupby(['mask_type', 'model', 'top_x_percent']).agg({
#     'size': ['mean', 'std', 'count'],
#     'size_percentage': ['mean', 'std']
# }).reset_index()
# summary.columns = ['mask_type', 'model', 'top_x_percent', 'mean_size', 'std_size', 'count', 'mean_size_percentage', 'std_size_percentage']

# # Print the summary
# print("\nSummary of Mask Sizes:")
# print(summary)

# # Save the summary to a CSV file

# # Plot the combined data
# plt.figure(figsize=(14, 8))
# sns.set(style="whitegrid")

# # Plotting the mean size
# sns.lineplot(
#     data=summary,
#     x='top_x_percent',
#     y='mean_size',
#     hue='mask_type',
#     style='model',
#     markers=True,
#     dashes=False,
#     palette='tab10'
# )

# plt.title('Mean Mask Size for Each Mask Type and Top X%')
# plt.xlabel('Top X% of Changed Weights')
# plt.ylabel('Mean Mask Size')
# plt.legend(title='Mask Type')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('./mask_size_plot_combined.png')
