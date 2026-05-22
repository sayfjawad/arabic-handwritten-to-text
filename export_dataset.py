#!/usr/bin/env python3
"""
Export usable OCR feedback pairs from the archive as a LLaMA-Factory
compatible JSONL dataset for Qwen2.5-VL QLoRA fine-tuning.

Each exported sample contains:
  - An RGB copy of the preprocessed image (archive processed.png → dataset/images/)
  - The ground truth text (human correction if provided, otherwise confirmed result)

Usability rules:
  - Job must have status=success and a feedback rating
  - Thumbs-up with no text change  → confirmed correct  (uses model result as ground truth)
  - Thumbs-down with text corrected → human correction   (uses corrected text as ground truth)
  - Thumbs-down with no correction  → weak signal        (skipped)

Usage:
    python export_dataset.py
    python export_dataset.py --out dataset/arabic_ocr.jsonl --val-split 0.1
    python export_dataset.py --archive ./archive --out dataset/arabic_ocr.jsonl --seed 42
"""

import argparse
import json
import os
import random
import sys

from PIL import Image

PROMPT = (
    "اقرأ النص العربي الموجود في الصورة واستخرج النص فقط. "
    "لا تشرح. لا تترجم. لا تضف أي شيء غير النص المكتوب."
)


def scan_archive(archive_dir: str) -> tuple[list[dict], dict]:
    counters = {
        "no_feedback": 0,
        "weak_signal": 0,
        "missing_image": 0,
        "empty_text": 0,
    }
    pairs = []

    if not os.path.isdir(archive_dir):
        print(f"[error] Archive directory not found: {archive_dir}", file=sys.stderr)
        sys.exit(1)

    for date_dir in sorted(os.listdir(archive_dir)):
        date_path = os.path.join(archive_dir, date_dir)
        if not os.path.isdir(date_path):
            continue

        for job_id in sorted(os.listdir(date_path)):
            job_dir   = os.path.join(date_path, job_id)
            meta_path = os.path.join(job_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            if meta.get("status") != "success":
                continue

            rating        = meta.get("feedback_rating")
            is_correction = meta.get("feedback_is_correction", False)

            if not rating:
                counters["no_feedback"] += 1
                continue

            if rating == "down" and not is_correction:
                counters["weak_signal"] += 1
                continue

            ground_truth = (
                meta.get("feedback_corrected_text", "") if is_correction
                else meta.get("result", "")
            )
            ground_truth = (ground_truth or "").strip()
            if not ground_truth:
                counters["empty_text"] += 1
                continue

            # Prefer processed.png (matches inference pipeline); fall back to input image
            processed_path = os.path.join(job_dir, "processed.png")
            ext            = meta.get("file_ext", ".jpg")
            input_path     = os.path.join(job_dir, f"input{ext}")

            if os.path.exists(processed_path):
                source_image = processed_path
            elif os.path.exists(input_path):
                source_image = input_path
            else:
                counters["missing_image"] += 1
                continue

            pairs.append({
                "job_id":        job_id,
                "date":          date_dir,
                "source_image":  source_image,
                "ground_truth":  ground_truth,
                "feedback_type": "correction" if is_correction else "confirmed",
            })

    return pairs, counters


def export_image(source_path: str, dest_path: str):
    """Load image, convert to RGB, save to dest_path."""
    img = Image.open(source_path).convert("RGB")
    img.save(dest_path, format="PNG", optimize=False)


def to_sharegpt(pair: dict) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": f"<image>\n{PROMPT}"},
            {"from": "gpt",   "value": pair["ground_truth"]},
        ],
        "images": [pair["exported_image"]],
    }


def write_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_dataset_info_snippet(train_path: str):
    abs_path = os.path.abspath(train_path)
    print(
        "\nAdd this entry to LLaMA-Factory/data/dataset_info.json:\n"
        "\n"
        '  "arabic_ocr_corrections": {\n'
        f'    "file_name": "{abs_path}",\n'
        '    "formatting": "sharegpt",\n'
        '    "columns": { "messages": "conversations", "images": "images" }\n'
        "  }"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export OCR feedback pairs as a LLaMA-Factory JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--archive",   default="./archive",
                        help="Path to the archive directory")
    parser.add_argument("--out", "-o", default="dataset/arabic_ocr.jsonl",
                        help="Output path for the training JSONL")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction to hold out as validation (0 to skip)")
    parser.add_argument("--val-out",   default=None,
                        help="Output path for validation JSONL (derived from --out if omitted)")
    parser.add_argument("--images-dir", default=None,
                        help="Directory to write exported RGB images (derived from --out if omitted)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for train/val split")
    args = parser.parse_args()

    # Derive defaults from --out
    out_stem, out_ext = os.path.splitext(args.out)
    if args.val_out is None:
        args.val_out = f"{out_stem}_val{out_ext}"
    if args.images_dir is None:
        args.images_dir = os.path.join(os.path.dirname(args.out) or "dataset", "images")

    # ── Scan ────────────────────────────────────────────────────────────────
    print(f"Scanning archive: {os.path.abspath(args.archive)}\n")
    pairs, counters = scan_archive(args.archive)

    total_scanned = (
        len(pairs)
        + counters["no_feedback"]
        + counters["weak_signal"]
        + counters["missing_image"]
        + counters["empty_text"]
    )
    confirmed   = sum(1 for p in pairs if p["feedback_type"] == "confirmed")
    corrections = sum(1 for p in pairs if p["feedback_type"] == "correction")

    print(f"{'Jobs scanned':<30} {total_scanned}")
    print(f"{'Usable pairs':<30} {len(pairs)}  ({confirmed} confirmed, {corrections} corrected)")
    print(f"{'Skipped — no feedback':<30} {counters['no_feedback']}")
    print(f"{'Skipped — weak signal':<30} {counters['weak_signal']}")
    print(f"{'Skipped — missing image':<30} {counters['missing_image']}")
    print(f"{'Skipped — empty text':<30} {counters['empty_text']}")

    if not pairs:
        print("\nNo usable pairs found — collect more feedback before exporting.")
        sys.exit(0)

    if len(pairs) < 50:
        print(
            f"\n[warning] Only {len(pairs)} pairs available. "
            "Fine-tuning with fewer than 50 pairs may have limited effect."
        )

    # ── Export images ────────────────────────────────────────────────────────
    os.makedirs(args.images_dir, exist_ok=True)
    print(f"\nExporting RGB images to: {os.path.abspath(args.images_dir)}")

    for pair in pairs:
        dest = os.path.join(args.images_dir, f"{pair['job_id']}.png")
        try:
            export_image(pair["source_image"], dest)
            pair["exported_image"] = os.path.abspath(dest)
        except Exception as e:
            print(f"  [warning] Could not export image for job {pair['job_id']}: {e}")
            pair["exported_image"] = None

    pairs = [p for p in pairs if p["exported_image"]]
    print(f"Exported {len(pairs)} images.")

    # ── Split ────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(pairs)

    if args.val_split > 0 and len(pairs) >= 2:
        n_val       = max(1, int(len(pairs) * args.val_split))
        val_pairs   = pairs[:n_val]
        train_pairs = pairs[n_val:]
    else:
        val_pairs   = []
        train_pairs = pairs

    # ── Write JSONL ──────────────────────────────────────────────────────────
    train_records = [to_sharegpt(p) for p in train_pairs]
    write_jsonl(train_records, args.out)

    print(f"\nTraining set  : {len(train_records):>4} pairs  →  {args.out}")

    if val_pairs:
        val_records = [to_sharegpt(p) for p in val_pairs]
        write_jsonl(val_records, args.val_out)
        print(f"Validation set: {len(val_records):>4} pairs  →  {args.val_out}")

    print_dataset_info_snippet(args.out)


if __name__ == "__main__":
    main()
