import os
import argparse
import subprocess
import json
import shlex
from tqdm import tqdm

import boto3
from botocore.exceptions import ClientError

EXPECTED_FILES = ['metrics-all.jsonl',
 'metrics.json',
 'task-000-arc_challenge-metrics.json',
 'task-000-arc_challenge-predictions.jsonl',
 'task-000-arc_challenge-recorded-inputs.jsonl',
 'task-000-arc_challenge-requests.jsonl',
 'task-001-arc_easy-metrics.json',
 'task-001-arc_easy-predictions.jsonl',
 'task-001-arc_easy-recorded-inputs.jsonl',
 'task-001-arc_easy-requests.jsonl',
 'task-002-boolq-metrics.json',
 'task-002-boolq-predictions.jsonl',
 'task-002-boolq-recorded-inputs.jsonl',
 'task-002-boolq-requests.jsonl',
 'task-003-csqa-metrics.json',
 'task-003-csqa-predictions.jsonl',
 'task-003-csqa-recorded-inputs.jsonl',
 'task-003-csqa-requests.jsonl',
 'task-004-hellaswag-metrics.json',
 'task-004-hellaswag-predictions.jsonl',
 'task-004-hellaswag-recorded-inputs.jsonl',
 'task-004-hellaswag-requests.jsonl',
 'task-005-openbookqa-metrics.json',
 'task-005-openbookqa-predictions.jsonl',
 'task-005-openbookqa-recorded-inputs.jsonl',
 'task-005-openbookqa-requests.jsonl',
 'task-006-piqa-metrics.json',
 'task-006-piqa-predictions.jsonl',
 'task-006-piqa-recorded-inputs.jsonl',
 'task-006-piqa-requests.jsonl',
 'task-007-socialiqa-metrics.json',
 'task-007-socialiqa-predictions.jsonl',
 'task-007-socialiqa-recorded-inputs.jsonl',
 'task-007-socialiqa-requests.jsonl',
 'task-008-winogrande-metrics.json',
 'task-008-winogrande-predictions.jsonl',
 'task-008-winogrande-recorded-inputs.jsonl',
 'task-008-winogrande-requests.jsonl',
 'task-009-mmlu_abstract_algebra-metrics.json',
 'task-009-mmlu_abstract_algebra-predictions.jsonl',
 'task-009-mmlu_abstract_algebra-recorded-inputs.jsonl',
 'task-009-mmlu_abstract_algebra-requests.jsonl',
 'task-010-mmlu_anatomy-metrics.json',
 'task-010-mmlu_anatomy-predictions.jsonl',
 'task-010-mmlu_anatomy-recorded-inputs.jsonl',
 'task-010-mmlu_anatomy-requests.jsonl',
 'task-011-mmlu_astronomy-metrics.json',
 'task-011-mmlu_astronomy-predictions.jsonl',
 'task-011-mmlu_astronomy-recorded-inputs.jsonl',
 'task-011-mmlu_astronomy-requests.jsonl',
 'task-012-mmlu_business_ethics-metrics.json',
 'task-012-mmlu_business_ethics-predictions.jsonl',
 'task-012-mmlu_business_ethics-recorded-inputs.jsonl',
 'task-012-mmlu_business_ethics-requests.jsonl',
 'task-013-mmlu_clinical_knowledge-metrics.json',
 'task-013-mmlu_clinical_knowledge-predictions.jsonl',
 'task-013-mmlu_clinical_knowledge-recorded-inputs.jsonl',
 'task-013-mmlu_clinical_knowledge-requests.jsonl',
 'task-014-mmlu_college_biology-metrics.json',
 'task-014-mmlu_college_biology-predictions.jsonl',
 'task-014-mmlu_college_biology-recorded-inputs.jsonl',
 'task-014-mmlu_college_biology-requests.jsonl',
 'task-015-mmlu_college_chemistry-metrics.json',
 'task-015-mmlu_college_chemistry-predictions.jsonl',
 'task-015-mmlu_college_chemistry-recorded-inputs.jsonl',
 'task-015-mmlu_college_chemistry-requests.jsonl',
 'task-016-mmlu_college_computer_science-metrics.json',
 'task-016-mmlu_college_computer_science-predictions.jsonl',
 'task-016-mmlu_college_computer_science-recorded-inputs.jsonl',
 'task-016-mmlu_college_computer_science-requests.jsonl',
 'task-017-mmlu_college_mathematics-metrics.json',
 'task-017-mmlu_college_mathematics-predictions.jsonl',
 'task-017-mmlu_college_mathematics-recorded-inputs.jsonl',
 'task-017-mmlu_college_mathematics-requests.jsonl',
 'task-018-mmlu_college_medicine-metrics.json',
 'task-018-mmlu_college_medicine-predictions.jsonl',
 'task-018-mmlu_college_medicine-recorded-inputs.jsonl',
 'task-018-mmlu_college_medicine-requests.jsonl',
 'task-019-mmlu_college_physics-metrics.json',
 'task-019-mmlu_college_physics-predictions.jsonl',
 'task-019-mmlu_college_physics-recorded-inputs.jsonl',
 'task-019-mmlu_college_physics-requests.jsonl',
 'task-020-mmlu_computer_security-metrics.json',
 'task-020-mmlu_computer_security-predictions.jsonl',
 'task-020-mmlu_computer_security-recorded-inputs.jsonl',
 'task-020-mmlu_computer_security-requests.jsonl',
 'task-021-mmlu_conceptual_physics-metrics.json',
 'task-021-mmlu_conceptual_physics-predictions.jsonl',
 'task-021-mmlu_conceptual_physics-recorded-inputs.jsonl',
 'task-021-mmlu_conceptual_physics-requests.jsonl',
 'task-022-mmlu_econometrics-metrics.json',
 'task-022-mmlu_econometrics-predictions.jsonl',
 'task-022-mmlu_econometrics-recorded-inputs.jsonl',
 'task-022-mmlu_econometrics-requests.jsonl',
 'task-023-mmlu_electrical_engineering-metrics.json',
 'task-023-mmlu_electrical_engineering-predictions.jsonl',
 'task-023-mmlu_electrical_engineering-recorded-inputs.jsonl',
 'task-023-mmlu_electrical_engineering-requests.jsonl',
 'task-024-mmlu_elementary_mathematics-metrics.json',
 'task-024-mmlu_elementary_mathematics-predictions.jsonl',
 'task-024-mmlu_elementary_mathematics-recorded-inputs.jsonl',
 'task-024-mmlu_elementary_mathematics-requests.jsonl',
 'task-025-mmlu_formal_logic-metrics.json',
 'task-025-mmlu_formal_logic-predictions.jsonl',
 'task-025-mmlu_formal_logic-recorded-inputs.jsonl',
 'task-025-mmlu_formal_logic-requests.jsonl',
 'task-026-mmlu_global_facts-metrics.json',
 'task-026-mmlu_global_facts-predictions.jsonl',
 'task-026-mmlu_global_facts-recorded-inputs.jsonl',
 'task-026-mmlu_global_facts-requests.jsonl',
 'task-027-mmlu_high_school_biology-metrics.json',
 'task-027-mmlu_high_school_biology-predictions.jsonl',
 'task-027-mmlu_high_school_biology-recorded-inputs.jsonl',
 'task-027-mmlu_high_school_biology-requests.jsonl',
 'task-028-mmlu_high_school_chemistry-metrics.json',
 'task-028-mmlu_high_school_chemistry-predictions.jsonl',
 'task-028-mmlu_high_school_chemistry-recorded-inputs.jsonl',
 'task-028-mmlu_high_school_chemistry-requests.jsonl',
 'task-029-mmlu_high_school_computer_science-metrics.json',
 'task-029-mmlu_high_school_computer_science-predictions.jsonl',
 'task-029-mmlu_high_school_computer_science-recorded-inputs.jsonl',
 'task-029-mmlu_high_school_computer_science-requests.jsonl',
 'task-030-mmlu_high_school_european_history-metrics.json',
 'task-030-mmlu_high_school_european_history-predictions.jsonl',
 'task-030-mmlu_high_school_european_history-recorded-inputs.jsonl',
 'task-030-mmlu_high_school_european_history-requests.jsonl',
 'task-031-mmlu_high_school_geography-metrics.json',
 'task-031-mmlu_high_school_geography-predictions.jsonl',
 'task-031-mmlu_high_school_geography-recorded-inputs.jsonl',
 'task-031-mmlu_high_school_geography-requests.jsonl',
 'task-032-mmlu_high_school_government_and_politics-metrics.json',
 'task-032-mmlu_high_school_government_and_politics-predictions.jsonl',
 'task-032-mmlu_high_school_government_and_politics-recorded-inputs.jsonl',
 'task-032-mmlu_high_school_government_and_politics-requests.jsonl',
 'task-033-mmlu_high_school_macroeconomics-metrics.json',
 'task-033-mmlu_high_school_macroeconomics-predictions.jsonl',
 'task-033-mmlu_high_school_macroeconomics-recorded-inputs.jsonl',
 'task-033-mmlu_high_school_macroeconomics-requests.jsonl',
 'task-034-mmlu_high_school_mathematics-metrics.json',
 'task-034-mmlu_high_school_mathematics-predictions.jsonl',
 'task-034-mmlu_high_school_mathematics-recorded-inputs.jsonl',
 'task-034-mmlu_high_school_mathematics-requests.jsonl',
 'task-035-mmlu_high_school_microeconomics-metrics.json',
 'task-035-mmlu_high_school_microeconomics-predictions.jsonl',
 'task-035-mmlu_high_school_microeconomics-recorded-inputs.jsonl',
 'task-035-mmlu_high_school_microeconomics-requests.jsonl',
 'task-036-mmlu_high_school_physics-metrics.json',
 'task-036-mmlu_high_school_physics-predictions.jsonl',
 'task-036-mmlu_high_school_physics-recorded-inputs.jsonl',
 'task-036-mmlu_high_school_physics-requests.jsonl',
 'task-037-mmlu_high_school_psychology-metrics.json',
 'task-037-mmlu_high_school_psychology-predictions.jsonl',
 'task-037-mmlu_high_school_psychology-recorded-inputs.jsonl',
 'task-037-mmlu_high_school_psychology-requests.jsonl',
 'task-038-mmlu_high_school_statistics-metrics.json',
 'task-038-mmlu_high_school_statistics-predictions.jsonl',
 'task-038-mmlu_high_school_statistics-recorded-inputs.jsonl',
 'task-038-mmlu_high_school_statistics-requests.jsonl',
 'task-039-mmlu_high_school_us_history-metrics.json',
 'task-039-mmlu_high_school_us_history-predictions.jsonl',
 'task-039-mmlu_high_school_us_history-recorded-inputs.jsonl',
 'task-039-mmlu_high_school_us_history-requests.jsonl',
 'task-040-mmlu_high_school_world_history-metrics.json',
 'task-040-mmlu_high_school_world_history-predictions.jsonl',
 'task-040-mmlu_high_school_world_history-recorded-inputs.jsonl',
 'task-040-mmlu_high_school_world_history-requests.jsonl',
 'task-041-mmlu_human_aging-metrics.json',
 'task-041-mmlu_human_aging-predictions.jsonl',
 'task-041-mmlu_human_aging-recorded-inputs.jsonl',
 'task-041-mmlu_human_aging-requests.jsonl',
 'task-042-mmlu_human_sexuality-metrics.json',
 'task-042-mmlu_human_sexuality-predictions.jsonl',
 'task-042-mmlu_human_sexuality-recorded-inputs.jsonl',
 'task-042-mmlu_human_sexuality-requests.jsonl',
 'task-043-mmlu_international_law-metrics.json',
 'task-043-mmlu_international_law-predictions.jsonl',
 'task-043-mmlu_international_law-recorded-inputs.jsonl',
 'task-043-mmlu_international_law-requests.jsonl',
 'task-044-mmlu_jurisprudence-metrics.json',
 'task-044-mmlu_jurisprudence-predictions.jsonl',
 'task-044-mmlu_jurisprudence-recorded-inputs.jsonl',
 'task-044-mmlu_jurisprudence-requests.jsonl',
 'task-045-mmlu_logical_fallacies-metrics.json',
 'task-045-mmlu_logical_fallacies-predictions.jsonl',
 'task-045-mmlu_logical_fallacies-recorded-inputs.jsonl',
 'task-045-mmlu_logical_fallacies-requests.jsonl',
 'task-046-mmlu_machine_learning-metrics.json',
 'task-046-mmlu_machine_learning-predictions.jsonl',
 'task-046-mmlu_machine_learning-recorded-inputs.jsonl',
 'task-046-mmlu_machine_learning-requests.jsonl',
 'task-047-mmlu_management-metrics.json',
 'task-047-mmlu_management-predictions.jsonl',
 'task-047-mmlu_management-recorded-inputs.jsonl',
 'task-047-mmlu_management-requests.jsonl',
 'task-048-mmlu_marketing-metrics.json',
 'task-048-mmlu_marketing-predictions.jsonl',
 'task-048-mmlu_marketing-recorded-inputs.jsonl',
 'task-048-mmlu_marketing-requests.jsonl',
 'task-049-mmlu_medical_genetics-metrics.json',
 'task-049-mmlu_medical_genetics-predictions.jsonl',
 'task-049-mmlu_medical_genetics-recorded-inputs.jsonl',
 'task-049-mmlu_medical_genetics-requests.jsonl',
 'task-050-mmlu_miscellaneous-metrics.json',
 'task-050-mmlu_miscellaneous-predictions.jsonl',
 'task-050-mmlu_miscellaneous-recorded-inputs.jsonl',
 'task-050-mmlu_miscellaneous-requests.jsonl',
 'task-051-mmlu_moral_disputes-metrics.json',
 'task-051-mmlu_moral_disputes-predictions.jsonl',
 'task-051-mmlu_moral_disputes-recorded-inputs.jsonl',
 'task-051-mmlu_moral_disputes-requests.jsonl',
 'task-052-mmlu_moral_scenarios-metrics.json',
 'task-052-mmlu_moral_scenarios-predictions.jsonl',
 'task-052-mmlu_moral_scenarios-recorded-inputs.jsonl',
 'task-052-mmlu_moral_scenarios-requests.jsonl',
 'task-053-mmlu_nutrition-metrics.json',
 'task-053-mmlu_nutrition-predictions.jsonl',
 'task-053-mmlu_nutrition-recorded-inputs.jsonl',
 'task-053-mmlu_nutrition-requests.jsonl',
 'task-054-mmlu_philosophy-metrics.json',
 'task-054-mmlu_philosophy-predictions.jsonl',
 'task-054-mmlu_philosophy-recorded-inputs.jsonl',
 'task-054-mmlu_philosophy-requests.jsonl',
 'task-055-mmlu_prehistory-metrics.json',
 'task-055-mmlu_prehistory-predictions.jsonl',
 'task-055-mmlu_prehistory-recorded-inputs.jsonl',
 'task-055-mmlu_prehistory-requests.jsonl',
 'task-056-mmlu_professional_accounting-metrics.json',
 'task-056-mmlu_professional_accounting-predictions.jsonl',
 'task-056-mmlu_professional_accounting-recorded-inputs.jsonl',
 'task-056-mmlu_professional_accounting-requests.jsonl',
 'task-057-mmlu_professional_law-metrics.json',
 'task-057-mmlu_professional_law-predictions.jsonl',
 'task-057-mmlu_professional_law-recorded-inputs.jsonl',
 'task-057-mmlu_professional_law-requests.jsonl',
 'task-058-mmlu_professional_medicine-metrics.json',
 'task-058-mmlu_professional_medicine-predictions.jsonl',
 'task-058-mmlu_professional_medicine-recorded-inputs.jsonl',
 'task-058-mmlu_professional_medicine-requests.jsonl',
 'task-059-mmlu_professional_psychology-metrics.json',
 'task-059-mmlu_professional_psychology-predictions.jsonl',
 'task-059-mmlu_professional_psychology-recorded-inputs.jsonl',
 'task-059-mmlu_professional_psychology-requests.jsonl',
 'task-060-mmlu_public_relations-metrics.json',
 'task-060-mmlu_public_relations-predictions.jsonl',
 'task-060-mmlu_public_relations-recorded-inputs.jsonl',
 'task-060-mmlu_public_relations-requests.jsonl',
 'task-061-mmlu_security_studies-metrics.json',
 'task-061-mmlu_security_studies-predictions.jsonl',
 'task-061-mmlu_security_studies-recorded-inputs.jsonl',
 'task-061-mmlu_security_studies-requests.jsonl',
 'task-062-mmlu_sociology-metrics.json',
 'task-062-mmlu_sociology-predictions.jsonl',
 'task-062-mmlu_sociology-recorded-inputs.jsonl',
 'task-062-mmlu_sociology-requests.jsonl',
 'task-063-mmlu_us_foreign_policy-metrics.json',
 'task-063-mmlu_us_foreign_policy-predictions.jsonl',
 'task-063-mmlu_us_foreign_policy-recorded-inputs.jsonl',
 'task-063-mmlu_us_foreign_policy-requests.jsonl',
 'task-064-mmlu_virology-metrics.json',
 'task-064-mmlu_virology-predictions.jsonl',
 'task-064-mmlu_virology-recorded-inputs.jsonl',
 'task-064-mmlu_virology-requests.jsonl',
 'task-065-mmlu_world_religions-metrics.json',
 'task-065-mmlu_world_religions-predictions.jsonl',
 'task-065-mmlu_world_religions-recorded-inputs.jsonl',
 'task-065-mmlu_world_religions-requests.jsonl']

def check_s3_path_exists(bucket_name, path):
    s3 = boto3.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=path)
        if 'Contents' in response:
            return True
        else:
            return False
    except ClientError as e:
        print(e)
        raise

def main(args):
    with open(args.checkpoint_data, "r") as f:
        checkpoints = [json.loads(line) for line in f]

    with open(args.tasks, "r") as f:
        tasks = [line.strip() for line in f]
    tasks_name = os.path.splitext(os.path.basename(args.tasks))[0]

    for checkpoint in checkpoints:
        checkpoint_name = checkpoint["model_name"]
        checkpoints_location = checkpoint["checkpoints_location"]
        revisions = checkpoint["revisions"]
        for revision in revisions:
            checkpoint_path = os.path.join(checkpoints_location, revision)

        
            checkpoint_args = f"--model {checkpoint_name} --model-args model_path={checkpoint_path} --revision {revision}"
            task_args = f"--task {' '.join(tasks)}"
            remote_dir_path = f"{args.remote_output_dir_prefix.rstrip('/')}/{checkpoint_name}/{revision}/{tasks_name}"

            # check if this path exists on s3
            remote_dir_path_stripped = remote_dir_path.replace("s3://", "", 1)
            bucket_name = remote_dir_path_stripped.split("/")[0]
            remote_dir_path_stripped = "/".join(remote_dir_path_stripped.split("/")[1:])
            if check_s3_path_exists(bucket_name, remote_dir_path_stripped):
                print(f"Path {remote_dir_path} already exists on s3")
                if args.check_expected_files:
                    any_missing = False
                    for expected_file in tqdm(EXPECTED_FILES, desc="Checking for expected files"):
                        file_path = os.path.join(remote_dir_path_stripped, expected_file)
                        if not check_s3_path_exists(bucket_name, file_path):
                            any_missing = True
                            break
                    if not any_missing:
                        print(f"Path {remote_dir_path} already exists on s3 and all expected files are present. Skipping.")
                        continue
                    else:
                        print(f"Path {remote_dir_path} already exists on s3 but some expected files are missing. Re-running.")
                else:
                    continue
                    

            remote_output_dir_args = f"--remote-output-dir {remote_dir_path}"
            all_args = " ".join([checkpoint_args, task_args, remote_output_dir_args])

            subprocess_args = shlex.split(f"oe-eval {all_args}") + args.run_args
            if args.dry_run:
                print(subprocess_args)
                return
            else:
                subprocess.run(subprocess_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_data", type=str, required=True, help="Path to a file containing jsonl data where each line lists model_name, checkpoints_location, revisions")
    parser.add_argument("--tasks", type=str, required=True, help="Path to a file containing listing task names, one per line")
    parser.add_argument("--remote_output_dir_prefix", type=str, required=True, help="Prefix for the remote output directory")
    parser.add_argument("--dry_run", action="store_true", help="If set, print the commands that would be run without executing them")
    parser.add_argument("--check_expected_files", action="store_true", help="If set, check if the expected files are present in the remote directory before running the job")
    parser.add_argument("run_args", nargs='*', help="Additional arguments to pass to the subprocess call")
    args = parser.parse_args()

    main(args)