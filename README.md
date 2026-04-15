# MMTR JSONL Evaluation Pipeline

This pipeline targets the flattened `MMTR.jsonl` benchmark format, where each line is a single sample. It is designed for a public, git-safe release and does not depend on the legacy `shapes/source_meta` structure.

## Repository Files

- `run_mmtr_pipeline.sh`: public entry script with environment-variable-based configuration
- `step1_infer_mmtr_azure.py`: multi-image inference
- `step2_clean_mmtr_prediction.py`: prediction cleaning
- `step3_eval_mmtr_llm.py`: metric computation and summary generation
- `common_mmtr.py`: shared utilities and metric helpers
- `requirements.txt`: minimal Python dependencies, excluding PyTorch

## Expected Dataset Layout

This pipeline follows the dataset layout used in the Hugging Face release:

```text
<dataset_dir>/
  MMTR.jsonl
  images/
```

Each JSONL row is expected to include:

- `sample_id`
- `doc_id`
- `context_img_paths`
- `file_name`
- `answer`
- `level`

Relative image paths are resolved against `DATASET_DIR`.

## Installation

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

### PyTorch is intentionally not included in `requirements.txt`

`step3_eval_mmtr_llm.py` uses `torch` and `sentence-transformers`, but `torch` is not pinned in `requirements.txt` because the correct build depends on the target machine, CUDA version, GPU driver, and installation method.

The original GPU environment used:

- `PyTorch 2.6.0+cu124`

Users should install PyTorch manually for their own environment before running evaluation. For example, for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Always verify the correct command from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your system.

## Configuration

This repository does not store secrets, internal service addresses, or machine-specific absolute paths.

Runtime configuration is provided through environment variables, including:

- `DATASET_DIR`
- `OUTPUT_BASE_DIR`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_BASE_URL`
- `AZURE_OPENAI_API_VERSION`
- `QWEN_EMBED_MODEL`
- `ENABLE_LLM_JUDGE`
- `LLM_JUDGE_API_KEY`
- `LLM_JUDGE_BASE_URL`
- `LLM_JUDGE_MODEL_NAME`

Recommended public workflow:

- keep the repository committed with placeholder-free, secret-free defaults
- inject real credentials through environment variables
- avoid committing private keys, internal endpoints, or local absolute paths

## Usage

Run the full pipeline:

```bash
DATASET_DIR=/path/to/mmtr_dataset bash run_mmtr_pipeline.sh
```

Run test mode on the first 10 samples:

```bash
DATASET_DIR=/path/to/mmtr_dataset bash run_mmtr_pipeline.sh test
```

By default, outputs are written to:

```text
<dataset_dir>/mmtr_eval_outputs/
```

## Output Files

- `mmtr_infer.jsonl`: inference output with `prediction`, `resolved_context_img_paths`, and `token_usage`
- `mmtr_clean.jsonl`: cleaned output with `raw_prediction` and `clean_prediction`
- `mmtr_eval.jsonl`: per-sample evaluation output with `scores`
- `mmtr_eval_summary.txt`: aggregated summary report

Test mode writes to a separate subdirectory and does not overwrite full-run results.

## Scoring Logic

- `L1`: `EM` + `ANLS`
- `L2-L4`: `Rouge-L` + `Qwen-EmbedSim`
- optional `LLM-Judge` binary gating

If the embedding model is not configured, the runner disables embedding evaluation automatically.

## Test Mode

- all three Python scripts support `--test_mode`
- `--test_mode` defaults to `--limit 10`
- if both `--limit` and `--test_mode` are provided, the explicit `--limit` takes precedence
