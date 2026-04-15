import argparse
import base64
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm

from common_mmtr import (
    ensure_parent_dir,
    load_jsonl,
    resolve_context_image_paths,
    save_jsonl,
    stable_sample_key,
)


DEFAULT_PROMPT = (
    "This is a document understanding benchmark task. "
    "The blacked-out regions in the images are synthetic masking artifacts introduced for evaluation, "
    "not intentional privacy redactions. "
    "Your task is to recover the masked text span using only visual evidence from the provided page images "
    "and their surrounding document context. "
    "Return the recovered text for the masked region only. "
    "If the span cannot be determined confidently, output [UNK]. "
    "Do not explain, refuse, or add any extra sentences."
)


def get_client(api_key: str, base_url: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=base_url,
    )


def get_image_mime_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def encode_image_raw(image_path: str) -> Tuple[str | None, int]:
    try:
        mime_type = get_image_mime_type(image_path)
        with open(image_path, "rb") as image_file:
            b64_str = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64_str}", len(b64_str)
    except Exception as exc:
        print(f"[Image Error] failed to encode raw image: {image_path} | {exc}")
        return None, 0


def encode_image_compressed(image_path: str, max_size: int = 1024, quality: int = 75) -> str | None:
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.thumbnail((max_size, max_size))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as exc:
        print(f"[Image Error] failed to compress image: {image_path} | {exc}")
        return None


def build_vision_payload(
    image_paths: List[str],
    max_payload_bytes: int,
    fallback_max_size: int,
    fallback_quality: int,
) -> List[str]:
    total_b64_size = 0
    raw_cache: List[Tuple[str, str]] = []
    valid_b64_list: List[str] = []

    for image_path in image_paths:
        if not os.path.exists(image_path):
            continue
        b64_str, b64_size = encode_image_raw(image_path)
        if b64_str:
            raw_cache.append((image_path, b64_str))
            total_b64_size += b64_size

    if not raw_cache:
        return []

    if total_b64_size <= max_payload_bytes:
        return [item[1] for item in raw_cache]

    img_count = len(raw_cache)
    dynamic_max_size = min(fallback_max_size, max(512, int(1500 / math.sqrt(img_count))))
    dynamic_quality = max(50, min(fallback_quality, 85 - (img_count * 2)))

    for image_path, _ in raw_cache:
        compressed = encode_image_compressed(
            image_path=image_path,
            max_size=dynamic_max_size,
            quality=dynamic_quality,
        )
        if compressed:
            valid_b64_list.append(compressed)

    return valid_b64_list


def call_gpt_blind_reconstruction(
    client: AzureOpenAI,
    model_name: str,
    reasoning_effort: str,
    prompt_text: str,
    b64_images_list: List[str],
) -> Tuple[str, Dict[str, int]]:
    content_payload: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for b64_img in b64_images_list:
        content_payload.append(
            {
                "type": "image_url",
                "image_url": {"url": b64_img},
            }
        )

    messages = [{"role": "user", "content": content_payload}]

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            content = response.choices[0].message.content
            if content is None:
                content = ""
            usage = getattr(response, "usage", None)
            token_usage = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
            return content.strip(), token_usage
        except Exception as exc:
            print(f"[API Error] attempt={attempt + 1} | {exc}")
            time.sleep(3)

    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def process_single_entry(
    item: Dict[str, Any],
    item_key: str,
    dataset_root: str,
    client: AzureOpenAI,
    model_name: str,
    reasoning_effort: str,
    prompt_text: str,
    max_payload_bytes: int,
    fallback_max_size: int,
    fallback_quality: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    resolved_image_paths = resolve_context_image_paths(item, dataset_root)
    valid_image_paths = [path for path in resolved_image_paths if os.path.exists(path)]

    out_item = dict(item)
    out_item["_cache_key"] = item_key
    out_item["resolved_context_img_paths"] = resolved_image_paths

    if not valid_image_paths:
        out_item["prediction"] = ""
        out_item["infer_meta"] = {
            "status": "missing_images",
            "num_input_images": len(resolved_image_paths),
            "num_existing_images": 0,
        }
        return out_item, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    valid_b64_list = build_vision_payload(
        image_paths=valid_image_paths,
        max_payload_bytes=max_payload_bytes,
        fallback_max_size=fallback_max_size,
        fallback_quality=fallback_quality,
    )

    if not valid_b64_list:
        out_item["prediction"] = ""
        out_item["infer_meta"] = {
            "status": "payload_build_failed",
            "num_input_images": len(resolved_image_paths),
            "num_existing_images": len(valid_image_paths),
        }
        return out_item, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prediction, token_usage = call_gpt_blind_reconstruction(
        client=client,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        prompt_text=prompt_text,
        b64_images_list=valid_b64_list,
    )

    out_item["prediction"] = prediction or ""
    out_item["infer_meta"] = {
        "status": "ok",
        "num_input_images": len(resolved_image_paths),
        "num_existing_images": len(valid_image_paths),
        "num_sent_images": len(valid_b64_list),
        "model_name": model_name,
        "reasoning_effort": reasoning_effort,
    }
    out_item["token_usage"] = token_usage
    return out_item, token_usage


def main() -> None:
    parser = argparse.ArgumentParser(description="MMTR jsonl inference pipeline for Azure OpenAI vision models")
    parser.add_argument("--input_jsonl", type=str, required=True, help="MMTR benchmark jsonl path")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Inference output jsonl path")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory used to resolve relative image paths. Default: input_jsonl parent directory.",
    )
    parser.add_argument("--model_name", type=str, default="gpt-5")
    parser.add_argument("--reasoning_effort", type=str, default="high", choices=["low", "medium", "high"])
    parser.add_argument("--api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", ""))
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("AZURE_OPENAI_BASE_URL", ""),
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=-1, help="Only process first N samples. -1 means all.")
    parser.add_argument("--test_mode", action="store_true", help="Run a quick test on the first 10 samples.")
    parser.add_argument("--empty_retry_rounds", type=int, default=2)
    parser.add_argument("--checkpoint_every", type=int, default=20)
    parser.add_argument("--max_payload_mb", type=float, default=8.0)
    parser.add_argument("--fallback_max_size", type=int, default=1024)
    parser.add_argument("--fallback_quality", type=int, default=75)
    parser.add_argument("--prompt_text", type=str, default=DEFAULT_PROMPT)
    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"Input jsonl not found: {args.input_jsonl}")
    if not args.api_key:
        raise ValueError("Please provide --api_key or set AZURE_OPENAI_API_KEY.")
    if not args.base_url:
        raise ValueError("Please provide --base_url or set AZURE_OPENAI_BASE_URL.")

    if args.test_mode and args.limit < 0:
        args.limit = 10
        print("Test mode enabled: only process first 10 samples.")

    dataset_root = args.dataset_root or os.path.dirname(os.path.abspath(args.input_jsonl))
    max_payload_bytes = int(args.max_payload_mb * 1024 * 1024)
    data = load_jsonl(args.input_jsonl, limit=args.limit)

    cached_results: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.output_jsonl):
        for idx, item in enumerate(load_jsonl(args.output_jsonl)):
            cached_results[stable_sample_key(item, idx)] = item

    indexed_data = []
    for idx, item in enumerate(data):
        item_key = stable_sample_key(item, idx)
        indexed_data.append((item_key, item))

    results_map: Dict[str, Dict[str, Any]] = dict(cached_results)
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    client = get_client(args.api_key, args.base_url, args.api_version)
    pending_items = []
    for item_key, item in indexed_data:
        cached = results_map.get(item_key)
        if cached and str(cached.get("prediction", "")).strip():
            continue
        pending_items.append((item_key, item))

    print(f"Loaded samples: {len(indexed_data)}")
    print(f"Cached finished: {len(indexed_data) - len(pending_items)}")
    print(f"Need inference: {len(pending_items)}")

    ensure_parent_dir(args.output_jsonl)
    processed_since_checkpoint = 0

    def flush_checkpoint() -> None:
        ordered = [results_map.get(item_key, dict(item, _cache_key=item_key)) for item_key, item in indexed_data]
        save_jsonl(args.output_jsonl, ordered)

    def run_one_round(round_items: List[Tuple[str, Dict[str, Any]]], round_id: int) -> None:
        nonlocal processed_since_checkpoint
        if not round_items:
            return

        print(f"Start round {round_id}: {len(round_items)} samples")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    process_single_entry,
                    item=item,
                    item_key=item_key,
                    dataset_root=dataset_root,
                    client=client,
                    model_name=args.model_name,
                    reasoning_effort=args.reasoning_effort,
                    prompt_text=args.prompt_text,
                    max_payload_bytes=max_payload_bytes,
                    fallback_max_size=args.fallback_max_size,
                    fallback_quality=args.fallback_quality,
                ): item_key
                for item_key, item in round_items
            }

            for future in tqdm(as_completed(future_map), total=len(future_map), desc=f"Infer round {round_id}"):
                result_item, usage = future.result()
                item_key = result_item["_cache_key"]
                results_map[item_key] = result_item
                token_stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                token_stats["completion_tokens"] += usage.get("completion_tokens", 0)
                token_stats["total_tokens"] += usage.get("total_tokens", 0)
                processed_since_checkpoint += 1
                if args.checkpoint_every > 0 and processed_since_checkpoint >= args.checkpoint_every:
                    flush_checkpoint()
                    processed_since_checkpoint = 0

        flush_checkpoint()
        processed_since_checkpoint = 0

    round_items = pending_items
    for round_id in range(1, args.empty_retry_rounds + 2):
        run_one_round(round_items, round_id)
        if round_id > args.empty_retry_rounds:
            break
        round_items = []
        for item_key, item in indexed_data:
            cached = results_map.get(item_key)
            if cached and not str(cached.get("prediction", "")).strip():
                round_items.append((item_key, item))
        if not round_items:
            break

    flush_checkpoint()

    token_txt = os.path.splitext(args.output_jsonl)[0] + "_token_usage.txt"
    with open(token_txt, "w", encoding="utf-8") as f:
        f.write("=== MMTR Inference Token Usage ===\n")
        f.write(f"Input: {args.input_jsonl}\n")
        f.write(f"Output: {args.output_jsonl}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Reasoning Effort: {args.reasoning_effort}\n")
        f.write(f"Prompt Tokens: {token_stats['prompt_tokens']}\n")
        f.write(f"Completion Tokens: {token_stats['completion_tokens']}\n")
        f.write(f"Total Tokens: {token_stats['total_tokens']}\n")

    empty_count = 0
    for item_key, item in indexed_data:
        cached = results_map.get(item_key, {})
        if not str(cached.get("prediction", "")).strip():
            empty_count += 1

    print(f"Saved inference jsonl: {args.output_jsonl}")
    print(f"Saved token report  : {token_txt}")
    print(f"Empty predictions   : {empty_count}")


if __name__ == "__main__":
    main()
