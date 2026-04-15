import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import editdistance


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str, limit: int = -1) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
            if limit > 0 and len(data) >= limit:
                break
    return data


def save_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def stable_sample_key(item: Dict[str, Any], idx: int) -> str:
    for key in ("sample_id", "id", "question_id", "uid"):
        value = item.get(key)
        if value not in (None, ""):
            return f"{key}::{value}"
    doc_id = item.get("doc_id", "")
    file_name = item.get("file_name", "")
    target_index = item.get("target_index", "")
    return f"doc::{doc_id}||file::{file_name}||target::{target_index}||idx::{idx}"


def canonical_level(raw_level: Any) -> str:
    if isinstance(raw_level, int):
        return f"L{raw_level}"
    text = str(raw_level).strip().upper()
    if text in {"1", "2", "3", "4"}:
        return f"L{text}"
    if text in {"L1", "L2", "L3", "L4"}:
        return text
    return "L3"


def get_prediction_text(item: Dict[str, Any]) -> str:
    for key in ("clean_prediction", "prediction", "pred", "raw_prediction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_ground_truth_text(item: Dict[str, Any]) -> str:
    answer = item.get("answer", "")
    return str(answer).strip() if answer is not None else ""


def normalize_extracted_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    wrappers = [
        r"^\s*extracted result\s*:\s*",
        r"^\s*final answer\s*:\s*",
        r"^\s*answer\s*:\s*",
        r"^\s*reconstructed text\s*:\s*",
        r"^\s*output\s*:\s*",
    ]
    for pattern in wrappers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"^\s*['\"](.*)['\"]\s*$", r"\1", text)
    return text.strip()


def resolve_context_image_paths(item: Dict[str, Any], dataset_root: str) -> List[str]:
    dataset_root = os.path.abspath(dataset_root)
    img_paths = []

    for rel_path in item.get("context_img_paths", []) or []:
        path = str(rel_path)
        if os.path.isabs(path):
            img_paths.append(path)
        else:
            img_paths.append(os.path.abspath(os.path.join(dataset_root, path)))

    if not img_paths:
        file_name = item.get("file_name")
        if file_name:
            if os.path.isabs(file_name):
                img_paths.append(file_name)
            else:
                img_paths.append(os.path.abspath(os.path.join(dataset_root, file_name)))

    deduped = []
    seen = set()
    for path in img_paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def normalize_span(text: str) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
    normalized = normalized.strip().lower()
    normalized = normalized.replace(",", "").replace("，", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def compute_em(pred: str, gt: str) -> float:
    pred_norm = normalize_span(pred)
    gt_norm = normalize_span(gt)
    if pred_norm == "" and gt_norm == "":
        return 1.0
    return 1.0 if pred_norm == gt_norm else 0.0


def levenshtein_similarity(pred: str, gt: str) -> float:
    pred_norm = normalize_span(pred)
    gt_norm = normalize_span(gt)
    if not pred_norm and not gt_norm:
        return 1.0
    if not pred_norm or not gt_norm:
        return 0.0
    dist = editdistance.eval(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))
    return 1.0 if max_len == 0 else 1.0 - (dist / max_len)


def compute_anls(pred: str, gt: str, theta: float = 0.5) -> float:
    similarity = levenshtein_similarity(pred, gt)
    return float(similarity if similarity >= theta else 0.0)


def compute_char_f1(pred: str, gt: str) -> float:
    pred = str(pred or "").strip()
    gt = str(gt or "").strip()
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pred_counter = Counter(pred)
    gt_counter = Counter(gt)
    common = sum((pred_counter & gt_counter).values())
    precision = common / len(pred)
    recall = common / len(gt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


PUNC_RE = re.compile(r"[，。！？、；：,.!?;:()\[\]{}<>\"'“”‘’/\\|]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
JP_RE = re.compile(r"[\u3040-\u30ff]")
KR_RE = re.compile(r"[\uac00-\ud7af]")


def tokenize_multilang(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = PUNC_RE.sub(" ", text).strip().lower()
    if not text:
        return []

    if " " in text and not (CJK_RE.search(text) or JP_RE.search(text) or KR_RE.search(text)):
        return [x for x in text.split() if x]

    out: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            out.append("".join(buf))
            buf.clear()

    for ch in text:
        if ch == " ":
            flush()
        elif CJK_RE.match(ch) or JP_RE.match(ch) or KR_RE.match(ch):
            flush()
            out.append(ch)
        else:
            buf.append(ch)

    flush()
    return [x for x in out if x]


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a
    prev = [0] * (len(short) + 1)
    cur = [0] * (len(short) + 1)
    for token in long_:
        cur[0] = 0
        for j in range(1, len(short) + 1):
            if token == short[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev, cur = cur, prev
    return prev[-1]


def compute_rouge_l_f1(pred: str, gt: str) -> float:
    pred_tokens = tokenize_multilang(pred)
    gt_tokens = tokenize_multilang(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gt_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    return 0.0 if (precision + recall) == 0 else float(2 * precision * recall / (precision + recall))
