# OlympiadBench Evaluation

## Sample Code for Evaluation
The `code` folder contains evaluation code used by OlympiadBench. It includes invocation of open-source/closed-source models to answer each problem, and evaluate those parts that can use automated scoring pipeline, in the end verifies its results. The experimental results in the paper are all obtained by running this code, though some prompts were slightly adjusted for publishing due to the modification of the name of dataset items.

#### Running
```
cd code/
# run evaluation
python evaluate_all.py --model_name <PATH_TO_MODEL> --dataset_name <DATASET_NAME>
# automated judging
cd ..
python judge.py
# print final results
python calculate_accuracy.py
```
