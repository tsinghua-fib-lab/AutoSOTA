# Evaluation

## Single Threaded Generation and evaluation

See slurm instructions below for how we parallelized the generation and evaluation.

### Prerequisites

    pip install -r requirements.txt

### First generate the gpt code

    python3 generate_gpt_codes.py -t /path/to/apps/test  --save /path/to/save_dir

### Second evaluate the accuracy of the outputted code

    python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir
    # because the above may fail on account of poorly generated python programs 
    # we suggest to run a for loop for each problem index against the "all_codes.json"
    for i in {0..#Num_Problems#} ; do 
      python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir -i $i ;
    done

The above will output the accuracy but to run it again once the evaluations have completed execute the line below:

    python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir --print_results

### Third evaluate the bleu scores of the outputted code

    python3 eval_bleu.py -t /path/to/apps/test --save /path/to/save_dir

Note: Third step does not depend on the second step.

## Parallelized Slurm Evaluation

Need to modify the path to apps in submit_all_jobs.sh to point to the evaluation folder and any other paths in that file as necessary. Install the requirements.txt file if you haven't already.

### Parrallel Generation and evaluation

    cd sbatch
    bash submit_all_jobs.sh

### Viewing the results

Once completed we provide a utility function to combine all of the smaller files into one larger file for ease of processing.

    python3 merge_codes.py --root /path/to/slurm/save_files.json

After the smaller files are combined we can view our accuracy with the following:

    python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir --print_results
