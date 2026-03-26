# CiteEval: Principle-Driven Citation Evaluation for Source Attribution
This repository releases the code and data for **CiteEval**:

> 📃 Title: **Principle-Driven Citation Evaluation for Source Attribution** <br>
> 🔗 Link: https://arxiv.org/pdf/2506.01829 <br>
> 🤔 Abstract: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations.

## Environment Setup
1. Set up Python environment
Create a virtualenv or conda environment with Python≥3.10

```bash
conda create -n citeeval python=3.10
conda activate citeeval
```

2. Install dependencies via:
```bash
pip install -r requirments.txt
```

3. Set up environment variables
```bash
# Set OPENAI_API_KEY 
export OPENAI_API_KEY='YOUR-OPENAI-API-KEY'

# Set CITEEVAL_ROOT
export CITEEVAL_ROOT="PATH-TO-CITEEVAL"

# Add CiteEval paths to `$PYTHONPATH`
export PYTHONPATH="${PYTHONPATH}:${CITEEVAL_PATH}:${CITEEVAL_PATH}/src/"
```

## Dataset Preparation: CiteBench

**Overview**
CiteBench consists of the following splits:

| Data Split             | Data Directory    | Query   |  Passages                     | Human Annotations|
| --------          | -------       |-------| ---------                     | ----------       |
| Metric Dev        | `metric_eval/metric_dev`  | YES    |    Retrieval      |        YES        |
| Metric Test       | `metric_eval/metric_test` | YES   |    Retrieval      |        YES       |
| Full Dev          | `dev`         | YES    | Retrieval+Oracle              |        NO       |
| Full Test         | `test`        | YES   |    Retrieval                  |        NO        |



**Detailed description**
- Human annotations are available for metric dev and metric test sets, ending with `human.out`.
- Queries in metric dev and test sets were sampled from the full dev set.
- Full dev and full test sets consist of separate files for each query data source: `asqa`, `eli5`, `msmarco`, and `lfrqa` (test only).
- Full dev set further includes oracle passages to study the effects of retrieval, ending with `*_oracle_dev.jsonl`.


**Steps to prepare**
1. Prepare data folders:
```
mkdir -p data/metric_eval data/metric_eval_outputs data/system_eval_outputs
```

2. Download CiteBench data from [Google Drive](https://drive.google.com/drive/folders/12Evj0f92wKz_7OGuuwq3KShTdSM8eu4v?usp=drive_link), and move it under `data/`.

**CiteBench Statistics**
You can check CiteBench statistics including IAA via: 

`python scripts/analyze_annotation.py --run iaa`

## Metric Evaluation
> This section provides instructions on performing metric evaluation (i.e., meta-evaluation) against human annotations in CiteBench.

We released metric output files for CiteEval-Auto and other comparsion metrics, under `data/metric_eval_outputs`.
You can obtain human correlation metrics for these metrics via: 
`sh run_metric_eval.sh`

To generate metric outputs for metric evaluation from scratch: `sh run_citeeval.sh`

## System Evaluation

> You can easily run CiteEval-Auto on your own system outputs, to assess the citation quality.

**Prerequisite**: Format your system output file as a `.json` file which includes an array of the following items: 
```json
{
    "id": "this is a sample id",
    "query": "this is a user query",
    "passages": [
        {
            "text": "this is the content of the first passage",
            "title": "this is the title of the document to which the first passage belongs (optional)"
        },
        {
            "text": "this is the content of the second passage",
            "title": "this is the title of the document to which the second passage belongs (optional)"
        }
    ],
    "pred": "this is a model-generated response with citations in brackets"
}
```
Move move file under `data/system_eval`, where we provide an example file `system_eval_examples.json`.

**Steps to benchmark with CiteEval-Auto**

1. Convert your system output file into CiteEval-compatible file using:
`python data/convert_to_citeeval_format.py --system_output_file $SYSEM_OUTPUT_FILE`
    - This generates a CiteEval input file ending with `.citeeval`, under the same directory

2. Generate CiteEval outputs with `sh run_citeeval.sh`.
    - In `run_citeeval.sh`, uncomment `SYSTEM EVAL` section and replace `PREDICTION_FILE` with your `.citeeval` file
    - This generates CiteEval metric output files ending with `.out`, under `data/metric_eval_outputs`

3. Print evaluation results with `sh run_system_eval.sh`. In `run_system_eval.sh`
    - Replace `PREDICTION_FILE` with the path to the `.citeeval` file
    - Add or remove `--cited` to switch between **Cited** / **Full** eval scenarios described in the paper


## Project Directory
- `src/modules`: core modules for CiteEval-Auto
- `src/scripts`: scripts for benchmark analysis, metric evaluation and system evaluation
- `src/data`: files for data creation and loading
- `src/common_utils`: common utils for logging and eval
- `src/template_configs`: prompt templates for CiteEval-Auto, including: 
    - `citeeval_ca.cfg`: context attribution
    - `citeeval_ce_cr.cfg`: citation editing and rating

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the CC-BY-NC-4.0 License.