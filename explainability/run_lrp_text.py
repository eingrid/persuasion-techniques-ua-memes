"""Token-level LRP (AttnLRP via lxt + zennit) on cached Qwen3-VL inference.

Loads per-sample records from `--inference_cache_dir/<mode>/`, computes input-embedding gradient
relevance on the predicted-label logits, normalises, and writes one JSON per sample to
`--output_dir/text_<mode>/`.

Usage:
    python run_lrp_text.py \\
        --model_id Qwen/Qwen3-VL-4B-Instruct \\
        --inference_cache_dir ./data_trans_qwen4/inference_cache \\
        --output_dir ./data_trans_qwen4/lrp_results \\
        --modes text image+text
"""
import argparse
import json
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import Dropout, LayerNorm
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP, Qwen3VLTextRMSNorm

from lxt.efficient import monkey_patch
from lxt.efficient.patches import (
    dropout_forward,
    gated_mlp_forward,
    layer_norm_forward,
    patch_attention,
    patch_method,
    rms_norm_forward,
)
from lxt.efficient.zennit_patches import monkey_patch_zennit


def install_lrp_patches():
    attn_lrp = {
        Qwen3VLTextMLP:    partial(patch_method, gated_mlp_forward),
        Qwen3VLTextRMSNorm: partial(patch_method, rms_norm_forward),
        Dropout:           partial(patch_method, dropout_forward),
        modeling_qwen3_vl: patch_attention,
        LayerNorm:         partial(patch_method, layer_norm_forward),
    }
    monkey_patch(modeling_qwen3_vl, patch_map=attn_lrp, verbose=True)
    monkey_patch_zennit(verbose=True)


def normalize_relevance(rel):
    rel = rel.clone().float()
    rel = torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    return rel / (rel.abs().max() + 1e-12)


def find_label_positions(seq, target_labels, input_len, tokenizer):
    positions = []
    for label in target_labels:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        for idx in range(input_len, len(seq) - len(label_ids) + 1):
            if seq[idx:idx + len(label_ids)] == label_ids:
                positions.extend(range(idx, idx + len(label_ids)))
                break
    return sorted(set(positions))


def compute_text_lrp(record, model, tokenizer):
    """Gradient×input relevance over the prompt tokens, summed over predicted-label logits."""
    output_ids = torch.tensor(record["output_ids"], dtype=torch.long).unsqueeze(0).to("cuda")
    input_len = record["input_len"]
    pred_labels = record.get("pred_labels", [])

    if not pred_labels:
        return None

    seq = record["output_ids"]
    label_positions = find_label_positions(seq, pred_labels, input_len, tokenizer)
    if not label_positions:
        return None

    logit_positions = [p - 1 for p in label_positions if p > 0]
    target_token_ids = [seq[p] for p in label_positions if p > 0]

    input_embeds = model.get_input_embeddings()(output_ids).detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=input_embeds,
        attention_mask=torch.ones_like(output_ids),
        use_cache=False,
    )
    target_logits = outputs.logits[0, logit_positions, target_token_ids]
    target_logits.sum().backward()

    relevance = (input_embeds * input_embeds.grad).float().sum(-1).detach().cpu()[0]
    relevance_norm = normalize_relevance(relevance)

    return {
        "token_ids":       record["output_ids"],
        "input_len":       input_len,
        "relevance_raw":   relevance.tolist(),
        "relevance_norm":  relevance_norm.tolist(),
        "label_positions": label_positions,
    }


def run_lrp_text(mode, cache_dir, out_root, model, tokenizer):
    mode_key = mode.replace("+", "_")
    mode_cache = cache_dir / mode_key
    out_dir = out_root / f"text_{mode_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((mode_cache / "manifest.json").read_text())
    failures = []
    n_ok = 0

    for entry in tqdm(manifest, desc=f"LRP text [{mode}]"):
        sample_id = entry["id"]
        out_file = out_dir / f"{sample_id}.json"
        if out_file.exists():
            n_ok += 1
            continue

        try:
            record = json.loads(Path(entry["path"]).read_text())
            lrp_result = compute_text_lrp(record, model, tokenizer)

            if lrp_result is None:
                failures.append({"id": sample_id, "error": "no labels or alignment"})
                continue

            lrp_result["id"] = sample_id
            lrp_result["mode"] = mode
            out_file.write_text(json.dumps(lrp_result, ensure_ascii=False, indent=2))
            n_ok += 1

        except Exception as e:
            print(f"  FAIL {sample_id}: {e}")
            traceback.print_exc()
            failures.append({"id": sample_id, "error": str(e)})
        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(f"Mode '{mode}': {n_ok} saved, {len(failures)} failures")


def parse_args():
    p = argparse.ArgumentParser(description="Token-level LRP over cached Qwen3-VL inference")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--inference_cache_dir", required=True,
                   help="Directory containing per-mode subdirs with manifest.json (output of run_inference.py).")
    p.add_argument("--output_dir", required=True,
                   help="LRP output root; per-mode subdirs `text_<mode>` are created.")
    p.add_argument("--modes", nargs="+", default=["text", "image+text"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cache_dir = Path(cfg.inference_cache_dir)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    install_lrp_patches()

    model = modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id, device_map="cuda", dtype=torch.bfloat16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    print(f"Model loaded: {cfg.model_id}")

    for mode in cfg.modes:
        run_lrp_text(mode, cache_dir, out_root, model, processor.tokenizer)


if __name__ == "__main__":
    main()
