#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from utils import (
    api_client,
    phase_probe_generator,
    probe_reasoning_agent,
    probe_responder,
    redteam_judge,
    task_plan,
)
from vision_utils import (
    typography_image_generator,
    vision_analyzer,
    vision_redteam_synthesizer,
)

# Vision Pipeline Runner Defaults (Including Typo/Stable Diffusion image generating, vision analysis)
DEFAULT_TYPOGRAPHY_MODEL = "deepseek-chat"
DEFAULT_TYPOGRAPHY_PROVIDER = api_client.get_provider_name_for_model(DEFAULT_TYPOGRAPHY_MODEL)
DEFAULT_TYPOGRAPHY_API_ENV_KEY = api_client.get_api_env_var_name_for_provider(DEFAULT_TYPOGRAPHY_PROVIDER)
DEFAULT_TYPOGRAPHY_BASE_URL = api_client.get_base_url_for_provider(DEFAULT_TYPOGRAPHY_PROVIDER)
DEFAULT_VISION_SETTINGS = api_client.get_agent_settings("probe_responder")
DEFAULT_VISION_MODEL = DEFAULT_VISION_SETTINGS.model
DEFAULT_VISION_PROVIDER = DEFAULT_VISION_SETTINGS.provider
DEFAULT_VISION_API_ENV_KEY = api_client.get_api_env_var_name_for_provider(DEFAULT_VISION_PROVIDER)
DEFAULT_VISION_BASE_URL = api_client.get_base_url_for_provider(DEFAULT_VISION_PROVIDER)
DEFAULT_VISION_SYNTHESIZER_SETTINGS = api_client.get_agent_settings("redteam_synthesizer")
DEFAULT_VISION_SYNTHESIZER_MODEL = DEFAULT_VISION_SYNTHESIZER_SETTINGS.model
DEFAULT_VISION_SYNTHESIZER_PROVIDER = DEFAULT_VISION_SYNTHESIZER_SETTINGS.provider
DEFAULT_VISION_SYNTHESIZER_API_ENV_KEY = api_client.get_api_env_var_name_for_provider(
    DEFAULT_VISION_SYNTHESIZER_PROVIDER
)
DEFAULT_VISION_SYNTHESIZER_BASE_URL = api_client.get_base_url_for_provider(
    DEFAULT_VISION_SYNTHESIZER_PROVIDER
)
DEFAULT_SD_MODEL_PATH = "/path/to/stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_SD_GUIDANCE_SCALE = 7.5
DEFAULT_SD_NUM_INFERENCE_STEPS = 50
DEFAULT_SD_WIDTH = 512
DEFAULT_SD_HEIGHT = 512


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json_atomic(path: str | Path, data: dict) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    temp_path.replace(target)
    return str(target)


def save_text(path: str | Path, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as file_obj:
        file_obj.write(content)
    return str(target)


def read_input_lines(input_path: str | Path) -> list[str]:
    with open(input_path, "r", encoding="utf-8") as file_obj:
        return [line.rstrip("\n") for line in file_obj]


def print_stage_header(title: str, details: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    for detail in details:
        print(f"  {detail}")
    print(f"{'=' * 60}")


def print_stage_completed(details: list[str]) -> None:
    print(f"\n{'=' * 60}")
    for detail in details:
        print(f"  {detail}")
    print(f"{'=' * 60}")


def build_typography_artifacts(
    input_path: str,
    output_root: str,
    typography_model: str = DEFAULT_TYPOGRAPHY_MODEL,
    typography_api_env_key: str = DEFAULT_TYPOGRAPHY_API_ENV_KEY,
    typography_base_url: str = DEFAULT_TYPOGRAPHY_BASE_URL,
    typography_system_prompt: str = typography_image_generator.DEFAULT_SYSTEM_PROMPT,
    typography_temperature: float = 0.4,
    typography_max_tokens: int = 120,
    image_size: int = typography_image_generator.DEFAULT_SIZE,
    font_size: int = typography_image_generator.DEFAULT_FONT_SIZE,
    font_path: str = typography_image_generator.DEFAULT_FONT_PATH,
    line_spacing: int = typography_image_generator.DEFAULT_LINE_SPACING,
    margin: int = typography_image_generator.DEFAULT_MARGIN,
) -> dict:
    input_name = Path(input_path).stem
    raw_lines = read_input_lines(input_path)
    def typography_fn(raw_text: str) -> str:
        return typography_image_generator.generate_typography(
            raw_question=raw_text,
            model=typography_model,
            api_env_key=typography_api_env_key,
            base_url=typography_base_url,
            system_prompt=typography_system_prompt,
            temperature=typography_temperature,
            max_tokens=typography_max_tokens,
        )

    records = typography_image_generator.build_typography_records(
        raw_lines,
        typography_fn=typography_fn,
    )
    rendered_records = typography_image_generator.render_typography_records(
        records,
        size=image_size,
        font_path=font_path,
        font_size=font_size,
        margin=margin,
        line_spacing=line_spacing,
    )

    image_dir = Path(output_root) / "typo_images" / input_name
    image_dir.mkdir(parents=True, exist_ok=True)

    for record in rendered_records:
        image = record["image"]
        image.save(image_dir / f"{record['index']}.png")

    text_json_path = Path(output_root) / "typography_texts" / f"{input_name}_typography_records.json"
    prompt_txt_path = Path(output_root) / "typography_texts" / f"{input_name}_vision_prompts.txt"

    save_json_atomic(
        text_json_path,
        {
            str(record["index"]): {
                "raw_text": record["raw_text"],
                "generated_text": record["generated_text"],
                "final_text": record["final_text"],
            }
            for record in rendered_records
        },
    )
    save_text(
        prompt_txt_path,
        "\n".join(f"{record['index']}|{record['final_text']}" for record in rendered_records),
    )

    return {
        "image_mode": "typo",
        "image_dir": str(image_dir),
        "typography_records_path": str(text_json_path),
        "typography_prompt_txt_path": str(prompt_txt_path),
    }


def build_sd_artifacts(
    input_path: str,
    output_root: str,
    typography_model: str = DEFAULT_TYPOGRAPHY_MODEL,
    typography_api_env_key: str = DEFAULT_TYPOGRAPHY_API_ENV_KEY,
    typography_base_url: str = DEFAULT_TYPOGRAPHY_BASE_URL,
    typography_system_prompt: str = typography_image_generator.DEFAULT_SYSTEM_PROMPT,
    typography_temperature: float = 0.4,
    typography_max_tokens: int = 120,
    sd_model_path: str = DEFAULT_SD_MODEL_PATH,
    guidance_scale: float = DEFAULT_SD_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_SD_NUM_INFERENCE_STEPS,
    width: int = DEFAULT_SD_WIDTH,
    height: int = DEFAULT_SD_HEIGHT,
) -> dict:
    from vision_utils import stable_diffusion_drawer

    typo_artifacts = build_typography_artifacts(
        input_path=input_path,
        output_root=output_root,
        typography_model=typography_model,
        typography_api_env_key=typography_api_env_key,
        typography_base_url=typography_base_url,
        typography_system_prompt=typography_system_prompt,
        typography_temperature=typography_temperature,
        typography_max_tokens=typography_max_tokens,
    )
    typo_records = load_json(typo_artifacts["typography_records_path"])

    prompts = [
        typo_records[key]["final_text"]
        for key in sorted((k for k in typo_records.keys() if k.isdigit()), key=int)
    ]

    sd_image_dir = Path(output_root) / "sd_images" / Path(input_path).stem
    sd_image_dir.mkdir(parents=True, exist_ok=True)

    pipeline = stable_diffusion_drawer.build_diffusion_pipeline(model_path=sd_model_path)
    sd_records = stable_diffusion_drawer.generate_images_from_prompts(
        prompts=prompts,
        pipeline=pipeline,
        output_dir=sd_image_dir,
        start_index=1,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        save_outputs=True,
    )

    serializable_records = {}
    for record in sd_records:
        serializable_records[str(record["index"])] = {
            "index": record["index"],
            "prompt": record["prompt"],
            "output_path": record["output_path"],
            "error": record["error"],
        }

    sd_records_path = Path(output_root) / "sd_images" / f"{Path(input_path).stem}_sd_records.json"
    save_json_atomic(sd_records_path, serializable_records)

    return {
        **typo_artifacts,
        "image_mode": "sd",
        "image_dir": str(sd_image_dir),
        "sd_records_path": str(sd_records_path),
    }


def build_vision_inputs(
    input_path: str,
    output_root: str,
    image_mode: str = "typo",
    typography_model: str = DEFAULT_TYPOGRAPHY_MODEL,
    typography_api_env_key: str = DEFAULT_TYPOGRAPHY_API_ENV_KEY,
    typography_base_url: str = DEFAULT_TYPOGRAPHY_BASE_URL,
    typography_system_prompt: str = typography_image_generator.DEFAULT_SYSTEM_PROMPT,
    typography_temperature: float = 0.4,
    typography_max_tokens: int = 120,
    sd_model_path: str = DEFAULT_SD_MODEL_PATH,
    sd_guidance_scale: float = DEFAULT_SD_GUIDANCE_SCALE,
    sd_num_inference_steps: int = DEFAULT_SD_NUM_INFERENCE_STEPS,
    sd_width: int = DEFAULT_SD_WIDTH,
    sd_height: int = DEFAULT_SD_HEIGHT,
) -> dict:
    if image_mode == "sd":
        return build_sd_artifacts(
            input_path=input_path,
            output_root=output_root,
            typography_model=typography_model,
            typography_api_env_key=typography_api_env_key,
            typography_base_url=typography_base_url,
            typography_system_prompt=typography_system_prompt,
            typography_temperature=typography_temperature,
            typography_max_tokens=typography_max_tokens,
            sd_model_path=sd_model_path,
            guidance_scale=sd_guidance_scale,
            num_inference_steps=sd_num_inference_steps,
            width=sd_width,
            height=sd_height,
        )

    return build_typography_artifacts(
        input_path=input_path,
        output_root=output_root,
        typography_model=typography_model,
        typography_api_env_key=typography_api_env_key,
        typography_base_url=typography_base_url,
        typography_system_prompt=typography_system_prompt,
        typography_temperature=typography_temperature,
        typography_max_tokens=typography_max_tokens,
    )


def run_vision_analysis(
    image_dir: str,
    output_root: str,
    vision_model: str,
    vision_api_env_key: str,
    vision_base_url: str,
    vision_temperature: float = 0.3,
    vision_max_tokens: int = 1024,
    vision_max_retries: int = 3,
) -> dict:
    client = vision_analyzer.build_client_from_env(
        api_env_key=vision_api_env_key,
        base_url=vision_base_url,
    )
    results = vision_analyzer.batch_analyze_vision(
        image_dir=image_dir,
        output_dir=str(Path(output_root) / "vision_inference"),
        client=client,
        model=vision_model,
        temperature=vision_temperature,
        max_tokens=vision_max_tokens,
        max_retries=vision_max_retries,
        save_output=True,
    )
    output_path = Path(output_root) / "vision_inference" / f"{Path(image_dir).name}_vision_inference.json"
    return {
        "vision_results": results,
        "vision_results_path": str(output_path),
    }


def resolve_vision_target_settings(
    responder_model: str,
    vision_model: str | None = None,
    vision_api_env_key: str | None = None,
    vision_base_url: str | None = None,
) -> tuple[str, str, str]:
    resolved_model = vision_model or responder_model
    resolved_settings = api_client.get_agent_settings(
        "probe_responder",
        model_override=resolved_model,
    )
    resolved_api_env_key = vision_api_env_key or api_client.get_api_env_var_name_for_provider(
        resolved_settings.provider
    )
    resolved_base_url = vision_base_url or api_client.get_base_url_for_provider(
        resolved_settings.provider
    )
    return resolved_settings.model, resolved_api_env_key, resolved_base_url


def repair_probe_outputs(
    breakdown_path: str,
    reasoning_path: str,
    probes_path: str,
    reasoning_model: str,
    optimizer_model: str,
    generator_model: str,
    verbose: bool = True,
) -> int:
    breakdowns = load_json(breakdown_path)
    reasonings = load_json(reasoning_path)
    probes = load_json(probes_path)

    repaired = 0

    for task_id, probe_record in probes.items():
        if task_id.startswith("_"):
            continue

        questions = probe_record.get("final_probe_questions", [])
        verdict = probe_record.get("verdict")
        needs_repair = verdict == "SKIP" or not questions

        if not needs_repair:
            continue

        task_data = breakdowns.get(task_id, {})
        threat_report = task_data.get("threat_analyst_report")
        original_request = task_data.get("original_request")

        if not threat_report or not original_request:
            continue

        if verbose:
            print(f"\n[Repair] Task {task_id} regenerating reasoning/probes")

        reasoning_result = probe_reasoning_agent.stage_r_reasoning(
            task_id=task_id,
            original_request=original_request,
            threat_report=threat_report,
            model=reasoning_model,
        )
        reasonings[task_id] = reasoning_result
        save_json_atomic(reasoning_path, reasonings)

        probe_result = phase_probe_generator.process_task(
            task_id=task_id,
            plan=reasoning_result,
            optimizer_model=optimizer_model,
            generator_model=generator_model,
            verbose=verbose,
        )
        probes[task_id] = probe_result["final_output"]
        save_json_atomic(probes_path, probes)
        repaired += 1

    return repaired


def run_vision_pipeline(
    input_path: str,
    image_mode: str = "typo",
    output_root: str = "Vision_Results",
    breakdown_dir: str = "Vision_Results/mission_breakdown_reports",
    reasoning_dir: str = "Vision_Results/probe_reasoning_results",
    probes_dir: str = "Vision_Results/phase_probe_results",
    responses_dir: str = "Vision_Results/probe_responses",
    synthesized_dir: str = "Vision_Results/vision_redteam_synthesized",
    judged_dir: str = "Vision_Results/redteam_judged",
    task_plan_model: str = "gpt-4o-mini",
    reasoning_model: str = "o4-mini",
    optimizer_model: str = "gpt-4o-mini",
    generator_model: str = "gpt-4o-mini",
    responder_model: str = "gpt-4o-mini",
    typography_model: str = DEFAULT_TYPOGRAPHY_MODEL,
    typography_api_env_key: str = DEFAULT_TYPOGRAPHY_API_ENV_KEY,
    typography_base_url: str = DEFAULT_TYPOGRAPHY_BASE_URL,
    typography_system_prompt: str = typography_image_generator.DEFAULT_SYSTEM_PROMPT,
    typography_temperature: float = 0.4,
    typography_max_tokens: int = 120,
    vision_model: str | None = None,
    vision_api_env_key: str | None = None,
    vision_base_url: str | None = None,
    vision_temperature: float = 0.3,
    vision_max_tokens: int = 1024,
    vision_max_retries: int = 3,
    vision_synthesizer_model: str = DEFAULT_VISION_SYNTHESIZER_MODEL,
    vision_synthesizer_api_env_key: str = DEFAULT_VISION_SYNTHESIZER_API_ENV_KEY,
    vision_synthesizer_base_url: str = DEFAULT_VISION_SYNTHESIZER_BASE_URL,
    vision_synthesizer_temperature: float = 0.3,
    vision_synthesizer_max_tokens: int = 8192,
    vision_synthesizer_max_retries: int = 3,
    vision_synthesizer_min_response_len: int = 100,
    judge_model: str = "gpt-4o-mini",
    sd_model_path: str = DEFAULT_SD_MODEL_PATH,
    sd_guidance_scale: float = DEFAULT_SD_GUIDANCE_SCALE,
    sd_num_inference_steps: int = DEFAULT_SD_NUM_INFERENCE_STEPS,
    sd_width: int = DEFAULT_SD_WIDTH,
    sd_height: int = DEFAULT_SD_HEIGHT,
    repair_skips: bool = True,
    verbose: bool = True,
) -> dict:
    resolved_vision_model, resolved_vision_api_env_key, resolved_vision_base_url = resolve_vision_target_settings(
        responder_model=responder_model,
        vision_model=vision_model,
        vision_api_env_key=vision_api_env_key,
        vision_base_url=vision_base_url,
    )
    total_inputs = len([line for line in read_input_lines(input_path) if line.strip()])
    vision_stage_title = (
        "Vision Typography Generator"
        if image_mode == "typo"
        else "Vision Typography + Stable Diffusion Generator"
    )
    print_stage_header(
        vision_stage_title,
        [
            f"Input file: {input_path}",
            f"Output root: {output_root}",
            f"Total tasks: {total_inputs} | Mode: {image_mode} | Model: {typography_model}",
        ],
    )
    vision_artifacts = build_vision_inputs(
        input_path=input_path,
        output_root=output_root,
        image_mode=image_mode,
        typography_model=typography_model,
        typography_api_env_key=typography_api_env_key,
        typography_base_url=typography_base_url,
        typography_system_prompt=typography_system_prompt,
        typography_temperature=typography_temperature,
        typography_max_tokens=typography_max_tokens,
        sd_model_path=sd_model_path,
        sd_guidance_scale=sd_guidance_scale,
        sd_num_inference_steps=sd_num_inference_steps,
        sd_width=sd_width,
        sd_height=sd_height,
    )
    print_stage_completed(
        [
            "Completed",
            f"Image directory: {vision_artifacts['image_dir']}",
            f"Typography records: {vision_artifacts.get('typography_records_path', '-')}",
        ]
        + (
            [f"Stable Diffusion records: {vision_artifacts['sd_records_path']}"]
            if "sd_records_path" in vision_artifacts
            else []
        )
    )

    print_stage_header(
        "Vision Analyzer",
        [
            f"Input directory: {vision_artifacts['image_dir']}",
            f"Output directory: {Path(output_root) / 'vision_inference'}",
            f"Total tasks: {total_inputs} | Model: {resolved_vision_model}",
        ],
    )
    vision_analysis = run_vision_analysis(
        image_dir=vision_artifacts["image_dir"],
        output_root=output_root,
        vision_model=resolved_vision_model,
        vision_api_env_key=resolved_vision_api_env_key,
        vision_base_url=resolved_vision_base_url,
        vision_temperature=vision_temperature,
        vision_max_tokens=vision_max_tokens,
        vision_max_retries=vision_max_retries,
    )
    print_stage_completed(
        [
            "Completed",
            f"Output file: {vision_analysis['vision_results_path']}",
        ]
    )

    breakdown_path = task_plan.batch_analyze_from_txt(
        input_txt_path=input_path,
        output_dir=breakdown_dir,
        model=task_plan_model,
    )
    reasoning_path = probe_reasoning_agent.process_batch(
        input_path=breakdown_path,
        output_dir=reasoning_dir,
        model=reasoning_model,
        verbose=verbose,
    )
    probes_path = phase_probe_generator.process_batch(
        input_path=reasoning_path,
        output_dir=probes_dir,
        optimizer_model=optimizer_model,
        generator_model=generator_model,
        verbose=verbose,
    )

    repaired = 0
    if repair_skips:
        repaired = repair_probe_outputs(
            breakdown_path=breakdown_path,
            reasoning_path=reasoning_path,
            probes_path=probes_path,
            reasoning_model=reasoning_model,
            optimizer_model=optimizer_model,
            generator_model=generator_model,
            verbose=verbose,
        )

    responses_path = probe_responder.process_batch(
        input_path=probes_path,
        output_dir=responses_dir,
        model=responder_model,
        verbose=verbose,
    )

    vision_category = Path(input_path).stem
    vision_synthesizer_client = vision_redteam_synthesizer.build_client_from_env(
        api_env_key=vision_synthesizer_api_env_key,
        base_url=vision_synthesizer_base_url,
    )
    synthesized_results = vision_redteam_synthesizer.process_batch(
        breakdown_path=breakdown_path,
        responses_path=responses_path,
        vision_path=vision_analysis["vision_results_path"],
        category_name=vision_category,
        output_dir=synthesized_dir,
        client=vision_synthesizer_client,
        model=vision_synthesizer_model,
        temperature=vision_synthesizer_temperature,
        max_tokens=vision_synthesizer_max_tokens,
        max_retries=vision_synthesizer_max_retries,
        min_response_len=vision_synthesizer_min_response_len,
        save_output=True,
    )
    synthesized_path = str(Path(synthesized_dir) / f"{Path(breakdown_path).stem}_synthesized_all.json")

    judged_path = redteam_judge.process_batch(
        input_path=synthesized_path,
        output_dir=judged_dir,
        model=judge_model,
        verbose=verbose,
    )

    return {
        "vision_image_mode": image_mode,
        "vision_image_dir": vision_artifacts["image_dir"],
        "vision_results_path": vision_analysis["vision_results_path"],
        "breakdown_path": breakdown_path,
        "reasoning_path": reasoning_path,
        "probes_path": probes_path,
        "responses_path": responses_path,
        "synthesized_path": synthesized_path,
        "judged_path": judged_path,
        "repaired_tasks": repaired,
        "vision_artifacts": vision_artifacts,
        "synthesized_results_meta": synthesized_results.get("_meta", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision pipeline runner")
    parser.add_argument("--input", "-i", required=True, help="A txt file with one task per line")
    parser.add_argument("--vision-image-mode", choices=["typo", "sd"], default="typo")
    parser.add_argument("--output-root", default="Vision_Results")
    parser.add_argument("--breakdown-dir", default="Vision_Results/mission_breakdown_reports")
    parser.add_argument("--reasoning-dir", default="Vision_Results/probe_reasoning_results")
    parser.add_argument("--probes-dir", default="Vision_Results/phase_probe_results")
    parser.add_argument("--responses-dir", default="Vision_Results/probe_responses")
    parser.add_argument("--synthesized-dir", default="Vision_Results/vision_redteam_synthesized")
    parser.add_argument("--judged-dir", default="Vision_Results/redteam_judged")
    parser.add_argument("--task-plan-model", default="gpt-4o-mini")
    parser.add_argument("--reasoning-model", default="o4-mini")
    parser.add_argument("--optimizer-model", default="gpt-4o-mini")
    parser.add_argument("--generator-model", default="gpt-4o-mini")
    parser.add_argument("--responder-model", default="gpt-4o-mini")
    parser.add_argument("--typography-model", default=DEFAULT_TYPOGRAPHY_MODEL)
    parser.add_argument("--typography-api-env-key", default=DEFAULT_TYPOGRAPHY_API_ENV_KEY)
    parser.add_argument("--typography-base-url", default=DEFAULT_TYPOGRAPHY_BASE_URL)
    parser.add_argument("--typography-system-prompt", default=typography_image_generator.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--typography-temperature", type=float, default=0.4)
    parser.add_argument("--typography-max-tokens", type=int, default=120)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--vision-api-env-key", default=None)
    parser.add_argument("--vision-base-url", default=None)
    parser.add_argument("--vision-temperature", type=float, default=0.3)
    parser.add_argument("--vision-max-tokens", type=int, default=1024)
    parser.add_argument("--vision-max-retries", type=int, default=3)
    parser.add_argument("--vision-synthesizer-model", default=DEFAULT_VISION_SYNTHESIZER_MODEL)
    parser.add_argument("--vision-synthesizer-api-env-key", default=DEFAULT_VISION_SYNTHESIZER_API_ENV_KEY)
    parser.add_argument("--vision-synthesizer-base-url", default=DEFAULT_VISION_SYNTHESIZER_BASE_URL)
    parser.add_argument("--vision-synthesizer-temperature", type=float, default=0.3)
    parser.add_argument("--vision-synthesizer-max-tokens", type=int, default=8192)
    parser.add_argument("--vision-synthesizer-max-retries", type=int, default=3)
    parser.add_argument("--vision-synthesizer-min-response-len", type=int, default=100)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--sd-model-path", default=DEFAULT_SD_MODEL_PATH)
    parser.add_argument("--sd-guidance-scale", type=float, default=DEFAULT_SD_GUIDANCE_SCALE)
    parser.add_argument("--sd-num-inference-steps", type=int, default=DEFAULT_SD_NUM_INFERENCE_STEPS)
    parser.add_argument("--sd-width", type=int, default=DEFAULT_SD_WIDTH)
    parser.add_argument("--sd-height", type=int, default=DEFAULT_SD_HEIGHT)
    parser.add_argument("--no-repair-skips", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_vision_pipeline(
        input_path=args.input,
        image_mode=args.vision_image_mode,
        output_root=args.output_root,
        breakdown_dir=args.breakdown_dir,
        reasoning_dir=args.reasoning_dir,
        probes_dir=args.probes_dir,
        responses_dir=args.responses_dir,
        synthesized_dir=args.synthesized_dir,
        judged_dir=args.judged_dir,
        task_plan_model=args.task_plan_model,
        reasoning_model=args.reasoning_model,
        optimizer_model=args.optimizer_model,
        generator_model=args.generator_model,
        responder_model=args.responder_model,
        typography_model=args.typography_model,
        typography_api_env_key=args.typography_api_env_key,
        typography_base_url=args.typography_base_url,
        typography_system_prompt=args.typography_system_prompt,
        typography_temperature=args.typography_temperature,
        typography_max_tokens=args.typography_max_tokens,
        vision_model=args.vision_model,
        vision_api_env_key=args.vision_api_env_key,
        vision_base_url=args.vision_base_url,
        vision_temperature=args.vision_temperature,
        vision_max_tokens=args.vision_max_tokens,
        vision_max_retries=args.vision_max_retries,
        vision_synthesizer_model=args.vision_synthesizer_model,
        vision_synthesizer_api_env_key=args.vision_synthesizer_api_env_key,
        vision_synthesizer_base_url=args.vision_synthesizer_base_url,
        vision_synthesizer_temperature=args.vision_synthesizer_temperature,
        vision_synthesizer_max_tokens=args.vision_synthesizer_max_tokens,
        vision_synthesizer_max_retries=args.vision_synthesizer_max_retries,
        vision_synthesizer_min_response_len=args.vision_synthesizer_min_response_len,
        judge_model=args.judge_model,
        sd_model_path=args.sd_model_path,
        sd_guidance_scale=args.sd_guidance_scale,
        sd_num_inference_steps=args.sd_num_inference_steps,
        sd_width=args.sd_width,
        sd_height=args.sd_height,
        repair_skips=not args.no_repair_skips,
        verbose=True,
    )

    print("\nVision pipeline outputs:")
    for key, value in result.items():
        print(f"  {key}: {value}")
