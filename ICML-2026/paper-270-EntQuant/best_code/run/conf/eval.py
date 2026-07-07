from hydra_zen import MISSING, store

from entquant.eval.eval_inference import BenchmarkConfig, EfficiencyModelEvaluator
from entquant.eval.eval_lm_eval import LMEvalModelEvaluator, TaskConfig
from entquant.eval.eval_ppl import PPLModelEvaluator
from entquant.eval.evaluator import ComposedModelEvaluator, ModelEvaluator
from run.hydra_zen import fbuilds

evaluator_ppl = fbuilds(
    PPLModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    dataset_names=[
        "c4",
        "wikitext2",
    ],
    ctx_length="${cfg.model.ctx_length}",
)
evaluator_lm_eval_base_full = fbuilds(
    LMEvalModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    tasks=[
        TaskConfig(name="piqa", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="openbookqa", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="lambada_openai", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="arc_easy", eval_kwargs={"num_fewshot": 25}),
        TaskConfig(name="boolq", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="truthfulqa_mc1", eval_kwargs={"num_fewshot": 0}),
        # Open LLM Leaderboard V1:
        TaskConfig(name="gsm8k", eval_kwargs={"num_fewshot": 5, "batch_size": 1}),
        TaskConfig(name="mmlu", eval_kwargs={"num_fewshot": 5}),
        TaskConfig(name="truthfulqa_mc2", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="hellaswag", instruct_mode=False, eval_kwargs={"num_fewshot": 10}),
        TaskConfig(name="winogrande", eval_kwargs={"num_fewshot": 5}),
        TaskConfig(name="arc_challenge", eval_kwargs={"num_fewshot": 25}),
    ],
    hflm_kwargs={"enable_thinking": False, "think_end_token": "</think>"},
)
evaluator_lm_eval_base = fbuilds(
    LMEvalModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    tasks=[
        TaskConfig(name="piqa", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="openbookqa", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="lambada_openai", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="arc_easy", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="boolq", eval_kwargs={"num_fewshot": 0, "batch_size": 1}),
        TaskConfig(name="truthfulqa_mc1", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="mmlu", eval_kwargs={"num_fewshot": 0, "batch_size": 1}),
        TaskConfig(name="truthfulqa_mc2", eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="hellaswag", instruct_mode=False, eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="winogrande", instruct_mode=False, eval_kwargs={"num_fewshot": 0}),
        TaskConfig(name="arc_challenge", eval_kwargs={"num_fewshot": 0}),
    ],
    hflm_kwargs={"enable_thinking": False, "think_end_token": "</think>"},
)
evaluator_lm_eval_adv_full = fbuilds(
    LMEvalModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    tasks=[
        TaskConfig(name="gsm8k_cot", eval_kwargs={"num_fewshot": 8}),
        TaskConfig(name="leaderboard"),  # Open LLM Leaderboard V2
    ],
    hflm_kwargs={"enable_thinking": False, "think_end_token": "</think>"},
)
tasks_adv = [
    TaskConfig(name="gsm8k_cot", eval_kwargs={"num_fewshot": 8, "batch_size": 1}),
    TaskConfig(name="gpqa_main_n_shot", eval_kwargs={"num_fewshot": 5, "batch_size": 1}),
    TaskConfig(name="mmlu", eval_kwargs={"num_fewshot": 5, "batch_size": 1}),
    TaskConfig(name="ifeval", eval_kwargs={"num_fewshot": 0}),
]
evaluator_lm_eval_adv = fbuilds(
    LMEvalModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    tasks=tasks_adv,
    hflm_kwargs={"enable_thinking": False, "think_end_token": "</think>"},
)
evaluator_lm_eval_adv_individual = {}
for task in tasks_adv:
    evaluator_lm_eval_adv_individual[task.name] = fbuilds(
        LMEvalModelEvaluator,
        tokenizer="${cfg.tokenizer}",
        tasks=[task],
        hflm_kwargs={"enable_thinking": False, "think_end_token": "</think>"},
    )
evaluator_efficiency = {}
PREFILL_BS = [1, 2, 4, 8, 16, 32]
PREFILL_SL = [512, 1024, 2048, 4096]
for bs in PREFILL_BS:
    for sl in PREFILL_SL:
        evaluator_efficiency[f"pre_bs{bs}_sl{sl}"] = fbuilds(
            EfficiencyModelEvaluator,
            tokenizer="${cfg.tokenizer}",
            config=fbuilds(
                BenchmarkConfig,
                eval_prefill=True,
                prefill_batch_size=bs,
                prefill_sequence_length=sl,
                prefill_num_warmup_steps=5,
                prefill_num_steps=20,
                eval_decode=False,
                seed=42,
            ),
            prefix=f"pre_bs{bs}_sl{sl}",
        )

DECODE_BS = [1, 4, 16, 32, 64]
DECODE_CTX = [1, 512, 2048]
DECODE_GEN = [64, 128, 256]
for bs in DECODE_BS:
    for ctx in DECODE_CTX:
        for gen in DECODE_GEN:
            evaluator_efficiency[f"dec_bs{bs}_ctx{ctx}_gen{gen}"] = fbuilds(
                EfficiencyModelEvaluator,
                tokenizer="${cfg.tokenizer}",
                config=fbuilds(
                    BenchmarkConfig,
                    eval_prefill=False,
                    eval_decode=True,
                    decode_batch_size=bs,
                    decode_context_length=ctx,
                    decode_num_tokens_to_generate=gen,
                    decode_num_warmup_steps=5,
                    decode_num_steps=20,
                    seed=42,
                ),
                prefix=f"dec_bs{bs}_ctx{ctx}_gen{gen}",
            )

evaluator_efficiency_cpu_offload = fbuilds(
    EfficiencyModelEvaluator,
    tokenizer="${cfg.tokenizer}",
    config=fbuilds(
        BenchmarkConfig,
        eval_prefill=True,
        prefill_batch_size=8,
        prefill_sequence_length=2048,
        prefill_num_warmup_steps=5,
        prefill_num_steps=20,
        eval_decode=True,
        decode_batch_size=4,
        decode_context_length=512,
        decode_num_tokens_to_generate=64,
        decode_num_warmup_steps=5,
        decode_num_steps=20,
        use_cpu_offload=True,
        seed=42,
    ),
    prefix="cpu_offload",
)

eval_none = fbuilds(ModelEvaluator)
eval_composed = fbuilds(
    ComposedModelEvaluator,
    evaluators=MISSING,
    hydra_convert="all",
)
eval_full = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
        lm_eval_base=evaluator_lm_eval_base_full,
        lm_eval_adv=evaluator_lm_eval_adv_full,
        efficiency=evaluator_efficiency,
    )
)
eval_accuracy_base_full = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
        lm_eval_base=evaluator_lm_eval_base_full,
    )
)
eval_accuracy_base = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
        lm_eval_base=evaluator_lm_eval_base,
    )
)
eval_accuracy_adv_full = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
        lm_eval_adv=evaluator_lm_eval_adv_full,
    )
)
eval_accuracy_adv = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
        lm_eval_adv=evaluator_lm_eval_adv,
    )
)
eval_accuracy_adv_individual = {}
for task in tasks_adv:
    eval_accuracy_adv_individual[task.name] = eval_composed(
        evaluators=dict(lm_eval_adv=evaluator_lm_eval_adv_individual[task.name])
    )
eval_ppl = eval_composed(
    evaluators=dict(
        ppl=evaluator_ppl,
    )
)
eval_efficiency = eval_composed(evaluators=evaluator_efficiency)
eval_efficiency_cpu_offload = eval_composed(evaluators=dict(cpu_offload=evaluator_efficiency_cpu_offload))

store(eval_none, group="cfg/eval", name="none")
store(eval_full, group="cfg/eval", name="full")
store(eval_accuracy_base_full, group="cfg/eval", name="accuracy_base_full")
store(eval_accuracy_base, group="cfg/eval", name="accuracy_base")
store(eval_accuracy_adv_full, group="cfg/eval", name="accuracy_adv_full")
store(eval_accuracy_adv, group="cfg/eval", name="accuracy_adv")
for task in tasks_adv:
    store(eval_accuracy_adv_individual[task.name], group="cfg/eval", name=f"accuracy_adv_{task.name}")
store(eval_ppl, group="cfg/eval", name="ppl")
store(eval_efficiency, group="cfg/eval", name="efficiency")
store(eval_efficiency_cpu_offload, group="cfg/eval", name="efficiency_cpu_offload")
