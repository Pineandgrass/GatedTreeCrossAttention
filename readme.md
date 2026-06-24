# GTCA: Gated-Tree Cross-Attention for Language Models

Code for the paper "Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs", presented at ACL 2026 main conference.

## Overview

GTCA augments decoder-only LLMs (Qwen-2.5-7B, Llama-3-8B) with a structural pathway that reads parse-tree chunk representations via gated cross-attention. At each transformer layer, a `ParseTreeCrossAttention` module attends to span-aligned chunk memory built from the constituency parse tree, with a causal chunk mask to preserve autoregressive behavior.

```
Input → Tokenizer → [Parse Tree Cache (SQLite)] → ParseTreeEncoder → chunk memory
                                                                          ↓
Token states → Self-Attention + ParseTreeCrossAttention (gated, causal) → Output
```

## Installation

```bash
conda env create -f environment.yml
```

## Datasets

Download from HuggingFace / GitHub and place under your `DATA_PATH`:

| Dataset | Source |
|---|---|
| HellaSwag | `Rowan/hellaswag` |
| WinoGrande | `allenai/winogrande` |
| MMLU | `cais/mmlu` |
| CLOTH | `AndyChiang/cloth` |
| BLiMP | [alexwarstadt/blimp](https://github.com/alexwarstadt/blimp) |
| CoLA (GLUE) | `nyu-mll/glue` |

## Generate parse-tree cache

```bash
python scripts/generate_tree_gtca.py \
  --model_name_or_path /path/to/qwen-2.5-7b \
  --task hellaswag \
  --data_path /path/to/data/hellaswag \
  --cache_path /path/to/cache/hellaswag_qwen.sqlite \
  --spacy_model en_core_web_sm \
  --benepar_model benepar_en3 \
  --splits train,validation,test \
  --include_label_splits train,validation
```

The cache is tokenizer-specific — regenerate for each backbone.

## Train (3-stage)

```bash
python scripts/train_gtca.py \
  --model_name_or_path /path/to/qwen-2.5-7b \
  --task mmlu \
  --data_path /path/to/data/mmlu \
  --cache_path /path/to/cache/mmlu_qwen.sqlite \
  --output_dir /path/to/outputs \
  --train_stage stage3 \
  --batch_size 1 --max_length 1024 --max_epochs 1 \
  --lr 1e-4 --alpha_max 1.0 --alpha_warmup_ratio 0.10 \
  --devices 1 --precision bf16-mixed
```

Training stages:
- `stage1` — LoRA only, GTCA frozen, alpha=0
- `stage2` — GTCA parameters only, LoRA frozen, alpha warmup
- `stage3` — Joint LoRA + GTCA, alpha fixed at `alpha_max`

## Evaluate

```bash
python scripts/test_gtca.py \
  --model_name_or_path /path/to/llama-3-8b \
  --checkpoint_path /path/to/checkpoint.ckpt \
  --task winogrande \
  --data_path /path/to/data/winogrande \
  --cache_path /path/to/cache/winogrande_llama.sqlite \
  --batch_size 1 --max_length 1024 --alpha 1.0 \
  --output_predictions /path/to/predictions.jsonl
```

## Citation
If you make use of the code in this repository, please cite the following papers:


```bibtex
@inproceedings{gao-etal-2026-gated,
    title = "Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only {LLM}s",
    author = "Gao, Xinyu  and
      Wang, Shaonan  and
      Ding, Nai",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1629/",
    pages = "35274--35288",
    ISBN = "979-8-89176-390-6"
}
