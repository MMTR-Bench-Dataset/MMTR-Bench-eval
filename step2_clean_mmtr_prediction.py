import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import AzureOpenAI, OpenAI
from tqdm import tqdm

from common_mmtr import (
    ensure_parent_dir,
    get_prediction_text,
    load_jsonl,
    normalize_extracted_text,
    save_jsonl,
    stable_sample_key,
)


DEFAULT_CLEAN_PROMPT = """You are cleaning OCR-style reconstruction outputs for a document benchmark.
Extract the final recovered text span from the raw model output.

Rules:
1. Keep only the reconstructed text span itself.
2. Remove explanations, prefixes like 'Answer:' or 'Final answer:', code fences, and surrounding quotes.
3. If the raw output is already a clean answer span, return it unchanged.
4. If the output expresses uncertainty such as unknown/not sure/cannot determine, return [UNK].
5. Output only the cleaned span, with no extra commentary.

Raw output:
{{RAW_PREDICTION}}"""


def render_prompt(prompt_template: str, raw_prediction: str) -> str:
    return prompt_template.replace("{{RAW_PREDICTION}}", raw_prediction)


def build_client(args: argparse.Namespace):
    if args.cleaner_backend == "rule":
        return None
    if args.cleaner_backend == "azure":
        if not args.api_key:
            raise ValueError("Please provide --api_key or set AZURE_OPENAI_API_KEY when using azure cleaner.")
        return AzureOpenAI(
            api_key=args.api_key,
            api_version=args.api_version,
            azure_endpoint=args.base_url,
        )
    if not args.api_key:
        raise ValueError("Please provide --api_key when using openai cleaner.")
    return OpenAI(api_key=args.api_key, base_url=args.base_url or None)


def call_cleaner_llm(client, backend: str, model_name: str, prompt: str, max_retries: int) -> str | None:
    for attempt in range(max_retries):
        try:
            if backend == "azure":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                return content.strip() if content else None

            response = client.responses.create(
                model=model_name,
                input=prompt,
            )
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            return None
        except Exception as exc:
            wait_sec = 2 ** attempt
            print(f"[Cleaner Error] {exc} | retry in {wait_sec}s")
            time.sleep(wait_sec)
    return None


def clean_prediction(item: Dict, item_key: str, args: argparse.Namespace, client, prompt_template: str) -> Dict:
    out_item = dict(item)
    out_item["_cache_key"] = item_key

    raw_prediction = str(item.get("prediction", "") or "")
    out_item["raw_prediction"] = raw_prediction

    rule_cleaned = normalize_extracted_text(raw_prediction)
    if args.cleaner_backend == "rule":
        cleaned = rule_cleaned
    else:
        if not raw_prediction.strip():
            cleaned = ""
        else:
            llm_prompt = render_prompt(prompt_template, raw_prediction)
            llm_cleaned = call_cleaner_llm(
                client=client,
                backend=args.cleaner_backend,
                model_name=args.model_name,
                prompt=llm_prompt,
                max_retries=args.max_retries,
            )
            cleaned = normalize_extracted_text(llm_cleaned if llm_cleaned else rule_cleaned)

    out_item["clean_prediction"] = cleaned
    out_item["prediction"] = cleaned
    out_item["clean_meta"] = {
        "backend": args.cleaner_backend,
        "model_name": args.model_name if args.cleaner_backend != "rule" else "rule_only",
    }
    return out_item


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean MMTR prediction jsonl produced by step1 inference")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--cleaner_backend", type=str, default="rule", choices=["rule", "openai", "azure"])
    parser.add_argument("--model_name", type=str, default="gpt-5-mini")
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument("--base_url", type=str, default=os.getenv("AZURE_OPENAI_BASE_URL", ""))
    parser.add_argument("--api_version", type=str, default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--test_mode", action="store_true", help="Run a quick test on the first 10 samples.")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--prompt_md", type=str, default=None, help="Optional prompt template markdown/text")
    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input jsonl not found: {args.input_jsonl}")

    if args.test_mode and args.limit < 0:
        args.limit = 10
        print("Test mode enabled: only process first 10 samples.")

    prompt_template = DEFAULT_CLEAN_PROMPT
    if args.prompt_md:
        with open(args.prompt_md, "r", encoding="utf-8") as f:
            prompt_template = f.read()

    data = load_jsonl(args.input_jsonl, limit=args.limit)
    cached_results: Dict[str, Dict] = {}
    if os.path.exists(args.output_jsonl):
        for idx, item in enumerate(load_jsonl(args.output_jsonl)):
            cached_results[stable_sample_key(item, idx)] = item

    indexed_data = []
    pending_items = []
    for idx, item in enumerate(data):
        item_key = stable_sample_key(item, idx)
        indexed_data.append((item_key, item))
        cached = cached_results.get(item_key)
        if cached and str(cached.get("clean_prediction", "")).strip():
            continue
        pending_items.append((item_key, item))

    print(f"Loaded samples: {len(indexed_data)}")
    print(f"Cached cleaned: {len(indexed_data) - len(pending_items)}")
    print(f"Need cleaning : {len(pending_items)}")

    client = build_client(args)
    results_map = dict(cached_results)
    ensure_parent_dir(args.output_jsonl)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                clean_prediction,
                item=item,
                item_key=item_key,
                args=args,
                client=client,
                prompt_template=prompt_template,
            ): item_key
            for item_key, item in pending_items
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Cleaning"):
            cleaned_item = future.result()
            results_map[cleaned_item["_cache_key"]] = cleaned_item

    ordered = [results_map.get(item_key, dict(item, _cache_key=item_key)) for item_key, item in indexed_data]
    save_jsonl(args.output_jsonl, ordered)
    print(f"Saved cleaned jsonl: {args.output_jsonl}")


if __name__ == "__main__":
    main()
