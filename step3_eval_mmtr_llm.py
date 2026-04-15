import argparse
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common_mmtr import (
    canonical_level,
    compute_anls,
    compute_char_f1,
    compute_em,
    compute_rouge_l_f1,
    ensure_parent_dir,
    get_ground_truth_text,
    get_prediction_text,
    load_jsonl,
    save_jsonl,
)


DEFAULT_QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "")
DEFAULT_LLM_JUDGE_LEVELS: Set[str] = {"L2", "L3", "L4"}

LEVEL_BASE_METRICS = {
    "L1": ["EM", "ANLS", "Char-F1"],
    "L2": ["Rouge-L", "Qwen-EmbedSim"],
    "L3": ["Rouge-L", "Qwen-EmbedSim"],
    "L4": ["Rouge-L", "Qwen-EmbedSim"],
}

CPU_METRICS = {"EM", "ANLS", "Char-F1", "Rouge-L"}
SEM_WEIGHT = {"L2": 0.30, "L3": 0.60, "L4": 0.80}

LEVEL_INSTRUCTIONS = {
    "L2": "Given a ground-truth short phrase and a reconstructed phrase, embed them so that semantically equivalent phrases are close even if wording differs.",
    "L3": "Given a ground-truth sentence and a reconstructed sentence, embed them for semantic equivalence evaluation while ignoring minor wording differences.",
    "L4": "Given a ground-truth paragraph and a reconstructed paragraph, embed them for long-text semantic similarity evaluation.",
}


def cpu_metric_worker(args: Tuple[int, str, str, str, List[str]]):
    sample_idx, pred_text, gt_text, _level, metrics = args
    scores: Dict[str, float] = {}
    for metric in metrics:
        if metric == "EM":
            scores[metric] = float(compute_em(pred_text, gt_text))
        elif metric == "ANLS":
            scores[metric] = float(compute_anls(pred_text, gt_text))
        elif metric == "Char-F1":
            scores[metric] = float(compute_char_f1(pred_text, gt_text))
        elif metric == "Rouge-L":
            scores[metric] = float(compute_rouge_l_f1(pred_text, gt_text))
    return sample_idx, scores


class QwenEmbedCalculator:
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f">>> init Qwen embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def compute_batch(self, level: str, preds: List[str], gts: List[str]) -> List[float]:
        if not preds:
            return []

        instruction = LEVEL_INSTRUCTIONS.get(level)
        sims = [0.0] * len(preds)
        valid_pairs = []

        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            pred = str(pred or "")
            gt = str(gt or "")
            if not pred and not gt:
                sims[idx] = 1.0
            elif not pred or not gt:
                sims[idx] = 0.0
            else:
                if instruction:
                    gt_text = f"Instruct: {instruction}\nText: {gt}"
                    pred_text = f"Instruct: {instruction}\nText: {pred}"
                else:
                    gt_text = gt
                    pred_text = pred
                valid_pairs.append((idx, pred_text, gt_text))

        if not valid_pairs:
            return sims

        pair_batch_size = max(1, self.batch_size // 2)
        with torch.inference_mode():
            for start in tqdm(range(0, len(valid_pairs), pair_batch_size), desc=f"Embedding {level}", leave=False):
                batch_pairs = valid_pairs[start:start + pair_batch_size]
                pred_batch = [x[1] for x in batch_pairs]
                gt_batch = [x[2] for x in batch_pairs]

                pred_embs = self.model.encode(
                    pred_batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=len(pred_batch),
                    normalize_embeddings=True,
                )
                gt_embs = self.model.encode(
                    gt_batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=len(gt_batch),
                    normalize_embeddings=True,
                )
                cos_scores = (pred_embs * gt_embs).sum(dim=1)
                for (pair_idx, _, _), score in zip(batch_pairs, cos_scores):
                    sims[pair_idx] = float(max(0.0, min(1.0, score.item())))

                del pred_embs, gt_embs, cos_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return sims

    def close(self) -> None:
        if hasattr(self, "model"):
            try:
                self.model.to("cpu")
            except Exception:
                pass
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def parse_binary_01(text: str) -> int:
    text = (text or "").strip()
    if text == "1":
        return 1
    if text == "0":
        return 0
    match = re.search(r"[01]", text)
    return int(match.group(0)) if match else 0


def call_llm_judge_binary(
    pred: str,
    gt: str,
    level: str,
    api_key: str,
    base_url: str,
    model_name: str,
    max_retries: int = 3,
) -> int:
    client = OpenAI(api_key=api_key, base_url=base_url)

    span_desc = {
        "L2": "short phrase",
        "L3": "sentence",
        "L4": "paragraph",
    }.get(level, "text span")

    system_prompt = "You are a strict binary evaluator. Output ONLY one character: 0 or 1. No explanation."
    user_prompt = f"""
[Task]
Decide whether the Prediction is a semantically equivalent reconstruction of the Ground Truth as a fill-in answer.

[Span type]
{span_desc}

[Ground Truth]
{gt}

[Prediction]
{pred}

[Decision rule]
Output 1 iff BOTH conditions hold:
1) Bidirectional entailment: Ground Truth entails Prediction AND Prediction entails Ground Truth.
2) No key modifier change, no missing key entities/numbers, no contradictions.

If the Prediction is only topic-related, partially correct, missing key info, changes key modifiers,
or introduces conflicting facts, output 0.

Output ONLY: 0 or 1
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.01 + (attempt * 0.05),
                max_tokens=2,
                stop=["\n", " "],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content
            if content and re.search(r"[01]", content):
                return parse_binary_01(content)
        except Exception as exc:
            print(f"[LLM Judge Error] {exc} on attempt {attempt + 1}/{max_retries}")
    return 0


def compute_final_score(item: Dict[str, Any], args: argparse.Namespace, qwen_calc: Optional[QwenEmbedCalculator]) -> None:
    scores = item.setdefault("scores", {})
    level = canonical_level(item.get("level"))
    pred_text = get_prediction_text(item)
    gt_text = get_ground_truth_text(item)

    if level == "L1":
        scores.setdefault("EM", float(compute_em(pred_text, gt_text)))
        scores.setdefault("ANLS", float(compute_anls(pred_text, gt_text)))
        scores.setdefault("Char-F1", float(compute_char_f1(pred_text, gt_text)))
        final = 0.7 * float(scores["EM"]) + 0.3 * float(scores["ANLS"])
        scores["Final-Score"] = float(max(0.0, min(1.0, final)))
        return

    scores.setdefault("Rouge-L", float(compute_rouge_l_f1(pred_text, gt_text)))

    if "Qwen-EmbedSim" not in scores:
        if qwen_calc is not None:
            scores["Qwen-EmbedSim"] = float(qwen_calc.compute_batch(level, [pred_text], [gt_text])[0])
        else:
            scores["Qwen-EmbedSim"] = float(scores["Rouge-L"])

    rouge_l = float(scores.get("Rouge-L", 0.0))
    embed_sim = float(scores.get("Qwen-EmbedSim", 0.0))
    weight = float(SEM_WEIGHT.get(level, 0.6))
    base_final = (1.0 - weight) * rouge_l + weight * embed_sim

    if args.enable_llm_judge and level in args.llm_judge_levels:
        if "LLM-Judge" not in scores:
            scores["LLM-Judge"] = float(
                call_llm_judge_binary(
                    pred=pred_text,
                    gt=gt_text,
                    level=level,
                    api_key=args.llm_api_key,
                    base_url=args.llm_base_url,
                    model_name=args.llm_model_name,
                )
            )
        judge = int(scores["LLM-Judge"])
        penalty = {"L2": 0.20, "L3": 0.30, "L4": 0.35}.get(level, 0.30)
        final = base_final * (penalty + (1.0 - penalty) * judge)
    else:
        final = base_final

    scores["Final-Score"] = float(max(0.0, min(1.0, final)))


def save_summary_report(output_jsonl: str, stats: Dict[str, Dict[str, List[float]]], sample_count: int) -> str:
    summary_path = os.path.splitext(output_jsonl)[0] + "_summary.txt"
    lines = []
    lines.append("=" * 72)
    lines.append("MMTR Benchmark Evaluation Report")
    lines.append(f"Source: {output_jsonl}")
    lines.append(f"Samples: {sample_count}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"{'Level':<8} | {'Metric':<16} | {'Mean(%)':>8} | {'Count':>6}")
    lines.append("-" * 72)

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    for level in sorted(stats.keys()):
        for metric, values in stats[level].items():
            mean_val = float(np.mean(values))
            lines.append(f"{level:<8} | {metric:<16} | {mean_val * 100:8.2f} | {len(values):6d}")
            metric_sums[metric] += sum(values)
            metric_counts[metric] += len(values)
        lines.append("-" * 72)

    lines.append("")
    lines.append("Overall")
    lines.append("-" * 72)
    lines.append(f"{'Metric':<16} | {'Mean(%)':>8} | {'Total Count':>10}")
    lines.append("-" * 72)
    for metric in ["Final-Score", "Qwen-EmbedSim", "Rouge-L", "EM", "ANLS", "Char-F1", "LLM-Judge"]:
        if metric_counts[metric] > 0:
            avg = metric_sums[metric] / metric_counts[metric]
            lines.append(f"{metric:<16} | {avg * 100:8.2f} | {metric_counts[metric]:10d}")
    lines.append("=" * 72)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f">>> summary saved to: {summary_path}")
    return summary_path


def parse_levels_csv(text: str) -> Set[str]:
    parts = [x.strip() for x in (text or "").split(",") if x.strip()]
    if not parts:
        return set()
    return {canonical_level(x) for x in parts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MMTR jsonl benchmark predictions")
    parser.add_argument("--input_jsonl", type=str, required=True, help="jsonl after cleaning")
    parser.add_argument("--output_jsonl", type=str, required=True, help="jsonl with scores")
    parser.add_argument("--qwen_model_name", type=str, default=DEFAULT_QWEN_EMBED_MODEL)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--qwen_batch_size", type=int, default=256)
    parser.add_argument("--disable_qwen_embed", action="store_true")
    parser.add_argument("--disable_llm_judge", action="store_true")
    parser.add_argument("--llm_judge_levels", type=str, default="L2,L3,L4")
    parser.add_argument("--llm_api_key", type=str, default=os.getenv("LLM_JUDGE_API_KEY", ""))
    parser.add_argument("--llm_base_url", type=str, default=os.getenv("LLM_JUDGE_BASE_URL", ""))
    parser.add_argument("--llm_model_name", type=str, default=os.getenv("LLM_JUDGE_MODEL_NAME", ""))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--test_mode", action="store_true", help="Run a quick test on the first 10 samples.")
    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input jsonl not found: {args.input_jsonl}")

    if args.test_mode and args.limit < 0:
        args.limit = 10
        print("Test mode enabled: only process first 10 samples.")

    args.enable_llm_judge = not args.disable_llm_judge
    args.llm_judge_levels = parse_levels_csv(args.llm_judge_levels) or DEFAULT_LLM_JUDGE_LEVELS

    if not args.disable_qwen_embed and not args.qwen_model_name:
        raise ValueError(
            "Please provide --qwen_model_name or set QWEN_EMBED_MODEL. "
            "If you do not want embedding evaluation, use --disable_qwen_embed."
        )

    if args.enable_llm_judge:
        missing_judge_args = []
        if not args.llm_api_key:
            missing_judge_args.append("--llm_api_key / LLM_JUDGE_API_KEY")
        if not args.llm_base_url:
            missing_judge_args.append("--llm_base_url / LLM_JUDGE_BASE_URL")
        if not args.llm_model_name:
            missing_judge_args.append("--llm_model_name / LLM_JUDGE_MODEL_NAME")
        if missing_judge_args:
            raise ValueError(
                "LLM Judge is enabled but missing configuration: " + ", ".join(missing_judge_args)
            )

    data = load_jsonl(args.input_jsonl, limit=args.limit)
    print(f"Loaded samples: {len(data)}")

    cpu_tasks = []
    qwen_tasks_by_level: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for sample_idx, item in enumerate(data):
        level = canonical_level(item.get("level"))
        pred_text = get_prediction_text(item)
        gt_text = get_ground_truth_text(item)
        scores = item.setdefault("scores", {})
        base_metrics = LEVEL_BASE_METRICS.get(level, LEVEL_BASE_METRICS["L3"])
        cpu_metric_list = [m for m in base_metrics if m in CPU_METRICS and m not in scores]
        need_qwen = "Qwen-EmbedSim" in base_metrics and "Qwen-EmbedSim" not in scores

        if cpu_metric_list:
            cpu_tasks.append((sample_idx, pred_text, gt_text, level, cpu_metric_list))
        if need_qwen and not args.disable_qwen_embed:
            qwen_tasks_by_level[level].append((sample_idx, pred_text, gt_text))

    if cpu_tasks:
        num_workers = args.num_workers if args.num_workers > 0 else max(1, cpu_count() - 1)
        print(f">>> computing lexical metrics with {num_workers} worker processes")
        with Pool(processes=num_workers) as pool:
            for sample_idx, scores in tqdm(pool.imap_unordered(cpu_metric_worker, cpu_tasks), total=len(cpu_tasks)):
                data[sample_idx].setdefault("scores", {}).update(scores)

    qwen_calc = None
    if qwen_tasks_by_level and not args.disable_qwen_embed:
        qwen_calc = QwenEmbedCalculator(args.qwen_model_name, args.device, args.qwen_batch_size)
        for level, tasks in qwen_tasks_by_level.items():
            preds = [task[1] for task in tasks]
            gts = [task[2] for task in tasks]
            sims = qwen_calc.compute_batch(level, preds, gts)
            for (sample_idx, _, _), sim in zip(tasks, sims):
                data[sample_idx].setdefault("scores", {})["Qwen-EmbedSim"] = float(sim)

    for item in tqdm(data, desc="Final scoring"):
        compute_final_score(item, args, qwen_calc)
        item["eval_meta"] = {
            "level": canonical_level(item.get("level")),
            "pred_field": "clean_prediction" if item.get("clean_prediction") is not None else "prediction",
        }

    if qwen_calc is not None:
        qwen_calc.close()

    ensure_parent_dir(args.output_jsonl)
    save_jsonl(args.output_jsonl, data)
    print(f">>> detailed results saved to: {args.output_jsonl}")

    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for item in data:
        level = canonical_level(item.get("level"))
        for metric, value in item.get("scores", {}).items():
            if value is not None:
                stats[level][metric].append(float(value))

    save_summary_report(args.output_jsonl, stats, sample_count=len(data))


if __name__ == "__main__":
    main()
