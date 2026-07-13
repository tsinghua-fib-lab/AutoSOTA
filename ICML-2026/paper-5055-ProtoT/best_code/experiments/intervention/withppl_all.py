import csv
import json
import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prototype_attn import ProtoBroadcastLM


LOAD_DIR_PROTO = "ProtoAttn_FineWeb_Large"
LOAD_DIR_SEED_124 = "trial000/seed_124"
LOAD_DIR_SEED_325 = "trial000/seed_325"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WRITE_MASK_VALUE = -float("inf")
READ_MASK_VALUE = -float("inf")
DEFAULT_OUTPUT_DIR = "withppl_results"

SentenceInput = Union[str, Sequence[str]]
TargetInput = Union[str, Sequence[str]]


INTERVENTION_MODE_CONFIGS = [
    {
        "name": "write_mask",
        "label": "WriteMask",
        "use_write_mask": True,
        "use_read_mask": False,
        "reinit_mode": "random",
    },
    {
        "name": "read_mask",
        "label": "ReadMask",
        "use_write_mask": False,
        "use_read_mask": True,
        "reinit_mode": "random",
    },
    {
        "name": "random",
        "label": "Random",
        "use_write_mask": False,
        "use_read_mask": False,
        "reinit_mode": "random",
    },
]

WOMEN_AND_GIRLS_SENTENCES = [
    "did you know that there is a government strategy for women and girls in sports and active recreation to address the inequalities of girls’ and women’s",
    "Many organizations are working on programs that focus on empowering women and girls to participate equally in science and technology.",
    "Did you know that several global initiatives aim to protect the rights of women and girls from violence and discrimination?",
    "Education policies are increasingly emphasizing equal opportunities for women and girls to excel in leadership roles.",
    "Access to healthcare remains a critical issue, and governments are creating strategies to improve services for women and girls.",
    "International campaigns highlight how climate change disproportionately affects women and girls in vulnerable communities.",
    "Did you know that mentorship networks are being created to support women and girls in pursuing careers in engineering and mathematics?",
]

MENTAL_SENTENCES = [
    "Meditation and mindfulness practices are beneficial for maintaining mental clarity.",
    "Regular exercise can improve mental well-being and reduce symptoms of anxiety.",
    "She sought professional help to manage her mental stress during the exam period.",
    "Many people face mental health challenges but do not seek support due to stigma.",
]

ZEALAND_SENTENCES = [
    "the dutch explorer abel tasman named new zealand as nova zeelandia after the dutch province of zeeland",
    "the treaty of waitangi signed in 1840 was instrumental in establishing british sovereignty over new zealand",
    "regular quaker meetings began in nelson in 1842 and later spread across new zealand",
    "the first quaker to visit aotearoa / new zealand was sydney parkinson",
]

COVID_SENTENCES = [
    "covid - 19 lambda variant lambda variant cases of covid - 19 are emerging in the us. while nowhere near",
    "The World Health Organization declared the outbreak of COVID-19 a pandemic in March 2020.",
    "Researchers identified the Alpha, Beta, Gamma, and Delta strains as variants of concern for COVID-19.",
    "The Pfizer-BioNTech and Moderna vaccines use mRNA technology to protect against the COVID-19 virus.",
    "Anosmia, the sudden loss of smell and taste, was identified as a specific symptom of COVID-19 infection.",
    "To curb the spread, the government mandated a 14-day quarantine for anyone testing positive for COVID-19.",
    "Hospitals faced a critical shortage of ventilators during the initial surge of severe COVID-19 cases.",
    "The FDA granted emergency use authorization for Paxlovid, an oral antiviral pill for treating COVID-19.",
    "Scientists continue to debate the zoonotic origins of COVID-19 and its potential transmission from bats.",
    "Despite strict border controls, the Omicron variant of COVID-19 managed to spread rapidly across the globe.",
    "Long-haulers are patients who suffer from debilitating symptoms months after recovering from acute COVID-19.",
    "Public health officials urged the population to wear N95 masks to prevent the airborne transmission of COVID-19.",
    "The CDC updated its guidelines regarding the isolation period for asymptomatic cases of COVID-19.",
    "Herd immunity against COVID-19 became difficult to achieve due to the emergence of new escape variants.",
    "Schools implemented social distancing and improved ventilation to reduce the risk of COVID-19 transmission in classrooms.",
    "The economic fallout from the COVID-19 pandemic led to supply chain disruptions and rising inflation.",
    "A negative PCR test result for COVID-19 was required for all passengers boarding international flights.",
    "Studies suggest that previous infection provides some level of natural immunity against reinfection with COVID-19.",
    "The global death toll attributed to COVID-19 has highlighted the vulnerabilities in healthcare systems worldwide.",
    "Contact tracing apps were deployed to alert individuals who had been exposed to a confirmed case of COVID-19.",
    "Rehabilitation programs are being established to help patients recover from the respiratory damage caused by severe COVID-19.",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    load_dir: str
    prototype_combos: Sequence[Tuple[int, int]]
    table_output_basename: str
    results_filename: str
    test_sentence: SentenceInput
    target_word: TargetInput
    output_dir: str = DEFAULT_OUTPUT_DIR


EXPERIMENTS = [
    ExperimentConfig(
        name="women",
        load_dir=LOAD_DIR_PROTO,
        prototype_combos=[(9, 2), (9, 18), (9, 7)],
        table_output_basename="results_step3_baseline_table_women_ppl",
        results_filename="results_step3_baseline_women_withppl.json",
        test_sentence=WOMEN_AND_GIRLS_SENTENCES,
        target_word=["women"],
    ),
    ExperimentConfig(
        name="girl",
        load_dir=LOAD_DIR_PROTO,
        prototype_combos=[(9, 2), (9, 18), (9, 7)],
        table_output_basename="results_step3_baseline_table_girl_ppl",
        results_filename="results_step3_baseline_girl_withppl.json",
        test_sentence=WOMEN_AND_GIRLS_SENTENCES,
        target_word=["girls"],
    ),
    ExperimentConfig(
        name="L6P9",
        load_dir=LOAD_DIR_SEED_124,
        prototype_combos=[(6, 9)],
        table_output_basename="results_inventations_124_L6P9_withppl_table",
        results_filename="results_step3_L6P9_withppl.json",
        test_sentence=MENTAL_SENTENCES,
        target_word=["mental"],
    ),
    ExperimentConfig(
        name="L5P9",
        load_dir=LOAD_DIR_SEED_124,
        prototype_combos=[(5, 9)],
        table_output_basename="results_inventations_124_L5P9_withppl_table",
        results_filename="results_step3_L5P9_withppl.json",
        test_sentence=ZEALAND_SENTENCES,
        target_word=["zealand"],
    ),
    ExperimentConfig(
        name="covid_baseline",
        load_dir=LOAD_DIR_PROTO,
        prototype_combos=[(1, 14)],
        table_output_basename="results_inventations_baseline_covid_withppl_table",
        results_filename="results_step3_covid_baseline_withppl.json",
        test_sentence=COVID_SENTENCES,
        target_word=["COVID"],
    ),
    ExperimentConfig(
        name="covid_325",
        load_dir=LOAD_DIR_SEED_325,
        prototype_combos=[(6, 31)],
        table_output_basename="results_inventations_325_covid_withppl_table",
        results_filename="results_step3_covid_325_withppl.json",
        test_sentence=COVID_SENTENCES,
        target_word=["COVID"],
    ),
    ExperimentConfig(
        name="covid_124",
        load_dir=LOAD_DIR_SEED_124,
        prototype_combos=[(7, 29)],
        table_output_basename="results_inventations_124_covid_withppl_table",
        results_filename="results_step3_covid_124_withppl.json",
        test_sentence=COVID_SENTENCES,
        target_word=["COVID"],
    ),
]


class FineWebBPETokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._setup_special_tokens()
        print(f"BPE Tokenizer loaded from {tokenizer_path}, vocab_size: {self.vocab_size}")

    def _setup_special_tokens(self):
        self.specials = {"<pad>": 0, "<eos>": 1, "<bos>": 1, "<sos>": 1}

    def get_pad_id(self):
        return self.specials["<pad>"]


def load_model_and_tokenizer(
    checkpoint_path: str,
    config_path: str,
    tokenizer_path: str,
    device: str = "cuda",
):
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    tokenizer = FineWebBPETokenizer(tokenizer_path)

    print(f"Model vocab: {model_config['vocab_size']}, Tokenizer vocab: {tokenizer.vocab_size}")
    assert model_config["vocab_size"] == tokenizer.vocab_size, "Vocab size mismatch!"

    model = ProtoBroadcastLM(
        vocab_size=model_config["vocab_size"],
        dim=model_config["dim"],
        depth=model_config["depth"],
        r=model_config["r"],
        max_seq_len=model_config["max_seq_len"],
        ffn_inner_size=model_config["ffn_inner_size"],
        dropout=0.0,
        pad_id=model_config.get("pad_id", tokenizer.get_pad_id()),
        tie_weights=model_config.get("tie_weights", True),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def get_target_word_probability(
    model: torch.nn.Module,
    tokenizer: FineWebBPETokenizer,
    sentence: str,
    target_word: str,
    device: str,
    use_future_context: bool = True,
) -> Tuple[float, int]:
    model.eval()
    sentence_tokens = tokenizer.tokenizer.encode(sentence).ids
    target_word_tokens = tokenizer.tokenizer.encode(target_word, add_special_tokens=False).ids

    if not target_word_tokens:
        raise ValueError(f"Target word '{target_word}' cannot be tokenized.")
    target_token_id = target_word_tokens[0]

    try:
        reversed_tokens = sentence_tokens[::-1]
        reversed_pos = reversed_tokens.index(target_token_id)
        target_pos = len(sentence_tokens) - 1 - reversed_pos
    except ValueError:
        raise ValueError(f"Target word '{target_word}' (ID: {target_token_id}) is not in the sentence.")

    if target_pos == 0:
        raise ValueError("Target word cannot be the first token in the sentence.")

    if use_future_context:
        input_tokens = sentence_tokens
    else:
        if target_pos <= 0:
            raise ValueError("Target word needs at least one preceding token for prefix probability.")
        input_tokens = sentence_tokens[:target_pos]

    input_ids = torch.tensor([input_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids, return_logits=True)
        if use_future_context:
            logits_for_target = logits[0, target_pos - 1, :]
        else:
            logits_for_target = logits[0, -1, :]
        probabilities = F.softmax(logits_for_target, dim=-1)
        target_probability = probabilities[target_token_id].item()

    return target_probability, target_token_id


def get_sentence_perplexity(
    model: torch.nn.Module,
    tokenizer: FineWebBPETokenizer,
    sentence: str,
    device: str,
) -> float:
    model.eval()
    tokens = tokenizer.tokenizer.encode(sentence).ids

    if len(tokens) < 2:
        return float("inf")

    ctx_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    input_ids = torch.tensor([ctx_tokens], dtype=torch.long).to(device)
    targets = torch.tensor([target_tokens], dtype=torch.long).to(device)
    pad_idx = tokenizer.get_pad_id()

    with torch.no_grad():
        logits = model(input_ids, return_logits=True)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_idx,
            reduction="mean",
        )
        return math.exp(loss.item()) if loss.item() > 0 else float("inf")


def truncate_sentence(text: str, max_len: int = 90) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def format_percentage(value: float) -> str:
    if value is None or value != value:
        return "-"
    return f"{value * 100:+.2f}%"


def export_table_reports(rows, mode_configs, combos, output_dir, table_output_basename):
    if not rows:
        return

    headers = ["Sentence", "Target", "Base Prob", "Base PPL"]
    for mode in mode_configs:
        for layer, proto in combos:
            headers.append(f"{mode['label']} L{layer}P{proto} (Prob Delta)")
            headers.append(f"{mode['label']} L{layer}P{proto} (PPL Delta)")

    table_data = []
    for row in rows:
        baseline_prob_str = f"{row['baseline_prob_prefix'] * 100:.2f}%"
        baseline_ppl_str = f"{row['baseline_ppl']:.2f}"

        cells = [truncate_sentence(row["sentence"]), row["target_word"], baseline_prob_str, baseline_ppl_str]
        mode_map = {m["mode"]: m for m in row["modes"]}

        for mode in mode_configs:
            results = mode_map.get(mode["name"], {}).get("results", [])
            proto_map = {(res["layer"], res["prototype"]): res for res in results}

            for combo in combos:
                res = proto_map.get(combo)
                if res is None:
                    cells.extend(["-", "-"])
                elif res.get("relative_change_prefix") is None:
                    cells.extend(["ERR" if res.get("error") else "-", "ERR" if res.get("error") else "-"])
                else:
                    cells.append(format_percentage(res["relative_change_prefix"]))
                    cells.append(format_percentage(res["relative_change_ppl"]))

        table_data.append(cells)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for cells in table_data:
        md_lines.append("| " + " | ".join(cells) + " |")

    md_path = os.path.join(output_dir, f"{table_output_basename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    csv_path = os.path.join(output_dir, f"{table_output_basename}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)

    print(f"Table (Markdown) saved to: {md_path}")
    print(f"Table (CSV) saved to: {csv_path}")


def get_clean_model_copy(model_original):
    model_copy = deepcopy(model_original)
    clear_model_masks(model_copy)
    return model_copy


def clear_model_masks(model):
    if hasattr(model, "clear_all_write_masks"):
        model.clear_all_write_masks()
    if hasattr(model, "clear_all_read_masks"):
        model.clear_all_read_masks()
    if hasattr(model, "backbone"):
        for block in model.backbone.blocks:
            if hasattr(block, "mixer"):
                if hasattr(block.mixer, "clear_write_mask"):
                    block.mixer.clear_write_mask()
                if hasattr(block.mixer, "clear_read_mask"):
                    block.mixer.clear_read_mask()


def run_interventions_for_mode(
    model_original,
    tokenizer,
    sentence,
    target_word,
    baseline_prob_prefix,
    baseline_ppl,
    mode_cfg,
    prototype_combos,
):
    results = []
    head_index = 0
    for layer_idx, proto_idx in prototype_combos:
        model_modified = get_clean_model_copy(model_original)
        status = "ok"
        error_msg = None
        sum_before = None
        sum_after = None
        mixer = None

        try:
            mixer = model_modified.backbone.blocks[layer_idx].mixer
            p_table = mixer.P_tables[head_index]
            sum_before = p_table[proto_idx].sum().item()

            if mode_cfg["use_write_mask"] and hasattr(mixer, "set_write_mask"):
                if hasattr(mixer, "clear_write_mask"):
                    mixer.clear_write_mask()
                if hasattr(mixer, "clear_read_mask"):
                    mixer.clear_read_mask()
                mask = torch.ones((mixer.cfg.num_heads, mixer.r_per_head), device=DEVICE)
                mask[head_index, proto_idx] = WRITE_MASK_VALUE
                mixer.set_write_mask(mask)
                sum_after = p_table[proto_idx].sum().item()
            elif mode_cfg["use_read_mask"] and hasattr(mixer, "set_read_mask"):
                if hasattr(mixer, "clear_write_mask"):
                    mixer.clear_write_mask()
                if hasattr(mixer, "clear_read_mask"):
                    mixer.clear_read_mask()
                mask = torch.ones((mixer.cfg.num_heads, mixer.r_per_head), device=DEVICE)
                mask[head_index, proto_idx] = READ_MASK_VALUE
                mixer.set_read_mask(mask)
                sum_after = p_table[proto_idx].sum().item()
            else:
                if hasattr(mixer, "clear_write_mask"):
                    mixer.clear_write_mask()
                if hasattr(mixer, "clear_read_mask"):
                    mixer.clear_read_mask()
                target_shape = p_table[proto_idx].shape
                if mode_cfg["reinit_mode"] == "zero":
                    new_vector = torch.zeros(target_shape, device=DEVICE)
                else:
                    channels = p_table.shape[-1]
                    new_vector = torch.randn(target_shape, device=DEVICE) / math.sqrt(channels)
                p_table.data[proto_idx] = new_vector
                sum_after = p_table[proto_idx].sum().item()

            modified_prob_prefix, _ = get_target_word_probability(
                model_modified, tokenizer, sentence, target_word, DEVICE, use_future_context=False
            )
            modified_ppl = get_sentence_perplexity(model_modified, tokenizer, sentence, DEVICE)

            if baseline_prob_prefix > 1e-12:
                relative_change_prefix = (modified_prob_prefix - baseline_prob_prefix) / baseline_prob_prefix
            else:
                relative_change_prefix = float("nan")

            if baseline_ppl > 0 and baseline_ppl != float("inf"):
                relative_change_ppl = (modified_ppl - baseline_ppl) / baseline_ppl
            else:
                relative_change_ppl = float("nan")

        except Exception as e:
            status = "error"
            error_msg = str(e)
            modified_prob_prefix = None
            relative_change_prefix = None
            modified_ppl = None
            relative_change_ppl = None
        finally:
            if mixer is not None:
                if hasattr(mixer, "clear_write_mask"):
                    mixer.clear_write_mask()
                if hasattr(mixer, "clear_read_mask"):
                    mixer.clear_read_mask()
            del model_modified

        results.append(
            {
                "layer": layer_idx,
                "prototype": proto_idx,
                "modified_prob_prefix": modified_prob_prefix,
                "relative_change_prefix": relative_change_prefix,
                "modified_ppl": modified_ppl,
                "relative_change_ppl": relative_change_ppl,
                "sum_before": sum_before,
                "sum_after": sum_after,
                "status": status,
                "error": error_msg,
            }
        )

    return results


def normalize_sentence_and_targets(test_sentence: SentenceInput, target_word: TargetInput):
    sentence_list = [test_sentence] if isinstance(test_sentence, str) else list(test_sentence)
    word_list = [target_word] if isinstance(target_word, str) else list(target_word)

    if len(sentence_list) != len(word_list):
        if len(sentence_list) == 1 and len(word_list) > 1:
            sentence_list = sentence_list * len(word_list)
        elif len(word_list) == 1 and len(sentence_list) > 1:
            word_list = word_list * len(sentence_list)
        else:
            raise ValueError("TEST_SENTENCE and TARGET_WORD lengths do not match and cannot be broadcast.")

    return sentence_list, word_list


def append_result_to_json(file_path: str, record: dict):
    data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
    data.append(record)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cached_model(load_dir: str, cache: Dict[str, Tuple[torch.nn.Module, FineWebBPETokenizer]]):
    if load_dir not in cache:
        model_state_dict_path = os.path.join(load_dir, "model_state_dict.pth")
        model_config_path = os.path.join(load_dir, "model_config.json")
        tokenizer_path = os.path.join(load_dir, "fineweb_bpe_16000.json")
        cache[load_dir] = load_model_and_tokenizer(
            model_state_dict_path,
            model_config_path,
            tokenizer_path,
            DEVICE,
        )
    return cache[load_dir]


def run_experiment(config: ExperimentConfig, model_cache, script_dir: str):
    print("\n" + "#" * 80)
    print(f"Running experiment: {config.name}")
    print("#" * 80)

    output_dir = os.path.join(script_dir, config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, config.results_filename)

    load_dir = config.load_dir if os.path.isabs(config.load_dir) else os.path.join(script_dir, config.load_dir)
    model_original, tokenizer = load_cached_model(load_dir, model_cache)
    sentence_list, word_list = normalize_sentence_and_targets(config.test_sentence, config.target_word)
    table_rows = []

    for cur_sentence, cur_word in zip(sentence_list, word_list):
        print("\n" + "=" * 60)
        print("STEP 3: ESTABLISHING BASELINE")
        print("=" * 60)
        try:
            baseline_prob_prefix, token_id = get_target_word_probability(
                model_original, tokenizer, cur_sentence, cur_word, DEVICE, use_future_context=False
            )
            baseline_ppl = get_sentence_perplexity(model_original, tokenizer, cur_sentence, DEVICE)

            if baseline_prob_prefix < 1e-6:
                print(
                    f"Baseline Probability (prefix) is too low: "
                    f"{baseline_prob_prefix:.6f} ({baseline_prob_prefix:.4%})"
                )
                continue
            print(f"Test Sentence: '{cur_sentence}'")
            print(f"Target Word:   '{cur_word}' (Token ID: {token_id})")
            print(f"Baseline Probability: {baseline_prob_prefix:.6f} ({baseline_prob_prefix:.4%})")
            print(f"Baseline PPL:         {baseline_ppl:.3f}")
        except ValueError as e:
            print(f"Error during baseline calculation: {e}")
            results_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "experiment": config.name,
                "sentence": cur_sentence,
                "target_word": cur_word,
                "error": str(e),
            }
            append_result_to_json(results_path, results_record)
            continue

        clear_model_masks(model_original)

        all_mode_results = []
        for mode_cfg in INTERVENTION_MODE_CONFIGS:
            print("\n" + "-" * 60)
            print(f"MODE: {mode_cfg['label']} (prefix only)")
            print("-" * 60)

            clear_model_masks(model_original)

            mode_results = run_interventions_for_mode(
                model_original,
                tokenizer,
                cur_sentence,
                cur_word,
                baseline_prob_prefix,
                baseline_ppl,
                mode_cfg,
                config.prototype_combos,
            )
            for result in mode_results:
                layer_idx = result["layer"]
                proto_idx = result["prototype"]
                modified_prob = result.get("modified_prob_prefix")
                modified_ppl = result.get("modified_ppl")

                if modified_prob is None or modified_ppl is None:
                    print(f"- L{layer_idx} P{proto_idx}: ERROR -> {result.get('error')}")
                else:
                    rel_prob = result.get("relative_change_prefix")
                    rel_ppl = result.get("relative_change_ppl")
                    rel_prob_str = f"{rel_prob:+.2%}" if isinstance(rel_prob, float) and rel_prob == rel_prob else "nan"
                    rel_ppl_str = f"{rel_ppl:+.2%}" if isinstance(rel_ppl, float) and rel_ppl == rel_ppl else "nan"
                    print(
                        f"- L{layer_idx} P{proto_idx}: "
                        f"Prob={modified_prob:.4%} (Delta {rel_prob_str}) | "
                        f"PPL={modified_ppl:.2f} (Delta {rel_ppl_str})"
                    )

            all_mode_results.append(
                {
                    "mode": mode_cfg["name"],
                    "label": mode_cfg["label"],
                    "results": mode_results,
                }
            )

        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS (ALL MODES)")
        print("=" * 60)
        print(f"Sentence: {cur_sentence}")
        print(f"Target:   {cur_word}")
        print(f"Baseline Prob: {baseline_prob_prefix:.6f} | Baseline PPL: {baseline_ppl:.3f}")

        for mode_entry in all_mode_results:
            print(f"\nMode: {mode_entry['label']}")
            for result in mode_entry["results"]:
                layer_idx = result["layer"]
                proto_idx = result["prototype"]
                if result.get("modified_prob_prefix") is not None:
                    rel_prob = result.get("relative_change_prefix")
                    rel_ppl = result.get("relative_change_ppl")
                    print(f"  - L{layer_idx} P{proto_idx}: Prob Delta {rel_prob:+.2%} | PPL Delta {rel_ppl:+.2%}")

        results_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment": config.name,
            "sentence": cur_sentence,
            "target_word": cur_word,
            "token_id": token_id,
            "baseline_prob_prefix": baseline_prob_prefix,
            "baseline_ppl": baseline_ppl,
            "modes": all_mode_results,
        }
        append_result_to_json(results_path, results_record)
        print(f"\nResults appended to: {results_path}")
        print("=" * 60)

        table_rows.append(
            {
                "sentence": cur_sentence,
                "target_word": cur_word,
                "baseline_prob_prefix": baseline_prob_prefix,
                "baseline_ppl": baseline_ppl,
                "modes": all_mode_results,
            }
        )

    export_table_reports(
        table_rows,
        INTERVENTION_MODE_CONFIGS,
        config.prototype_combos,
        output_dir,
        config.table_output_basename,
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_cache = {}
    for config in EXPERIMENTS:
        run_experiment(config, model_cache, script_dir)


if __name__ == "__main__":
    main()
