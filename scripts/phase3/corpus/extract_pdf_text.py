#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import fitz  # PyMuPDF


def safe_stem(name: str) -> str:
    # ファイル名として危険な文字を潰す
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = name.strip()
    return name or "noname"


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_text_and_blocks(pdf_path: Path, out_dir: Path, also_blocks: bool) -> dict:
    # Guard: skip empty / invalid files so batch runs don't crash.
    try:
        if pdf_path.exists() and pdf_path.stat().st_size == 0:
            return {
                "pdf": str(pdf_path),
                "status": "EMPTY_FILE",
                "error": "0-byte PDF",
                "txt": None,
                "blocks": None,
                "total_pages": 0,
                "low_text_pages": 0,
            }
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return {
            "pdf": str(pdf_path),
            "status": "OPEN_ERROR",
            "error": f"{type(e).__name__}: {e}",
            "txt": None,
            "blocks": None,
            "total_pages": 0,
            "low_text_pages": 0,
        }

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_stem(pdf_path.stem)
    out_txt = out_dir / f"{stem}.txt"
    out_blocks = out_dir / f"{stem}_blocks.csv"

    parts: list[str] = []
    low_text_pages = 0
    total_pages = len(doc)

    # blocks csv writer（必要なら）
    blocks_f = None
    blocks_w = None
    if also_blocks:
        blocks_f = open(out_blocks, "w", newline="", encoding="utf-8")
        blocks_w = csv.writer(blocks_f)
        blocks_w.writerow(["page", "block_id", "x0", "y0", "x1", "y1", "text"])

    try:
        for pno in range(total_pages):
            page = doc[pno]
            text = (page.get_text("text") or "").strip()

            if len(text) < 50:
                low_text_pages += 1
                parts.append(f"\n\n===== Page {pno+1} (LIKELY_IMAGE_OR_NO_TEXT) =====\n")
            else:
                parts.append(f"\n\n===== Page {pno+1} =====\n")
                parts.append(text)

            if also_blocks and blocks_w is not None:
                blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
                for b in blocks:
                    x0, y0, x1, y1, btxt, block_no, block_type = b
                    btxt = (btxt or "").strip()
                    if not btxt:
                        continue
                    blocks_w.writerow([pno + 1, block_no, x0, y0, x1, y1, btxt])
    finally:
        doc.close()
        if blocks_f is not None:
            blocks_f.close()

    out_txt.write_text(clean_text("\n".join(parts)), encoding="utf-8")
    return {
        "pdf": str(pdf_path),
        "status": "OK",
        "error": "",
        "txt": str(out_txt),
        "blocks": str(out_blocks) if also_blocks else None,
        "total_pages": total_pages,
        "low_text_pages": low_text_pages,
    }


def iter_pdfs(input_path: Path | None, input_dir: Path | None) -> list[Path]:
    if input_path:
        return [input_path]
    if input_dir:
        return sorted([p for p in input_dir.glob("*.pdf") if p.is_file()])
    raise ValueError("Either --input or --input_dir must be provided.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Single PDF path")
    ap.add_argument("--input_dir", type=str, default=None, help="Directory containing PDFs")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for txt/csv")
    ap.add_argument("--also_blocks", action="store_true", help="Also export blocks CSV")
    ap.add_argument("--skip_existing", action="store_true", help="Skip extraction if output files already exist")
    ap.add_argument("--append_summary", action="store_true", help="Append to _extract_summary.csv instead of overwriting")
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else None
    input_dir = Path(args.input_dir) if args.input_dir else None
    out_dir = Path(args.out_dir)

    pdfs = iter_pdfs(input_path, input_dir)
    if not pdfs:
        print("No PDFs found.")
        return

    summary = []
    for pdf in pdfs:
        stem = safe_stem(pdf.stem)
        out_txt = out_dir / f"{stem}.txt"
        out_blocks = out_dir / f"{stem}_blocks.csv"

        if args.skip_existing and out_txt.exists() and (not args.also_blocks or out_blocks.exists()):
            print(f"[SKIP] {pdf.name} (outputs exist)")
            summary.append({
                "pdf": str(pdf),
                "status": "SKIP_EXISTING",
                "error": "outputs exist",
                "txt": str(out_txt),
                "blocks": str(out_blocks) if args.also_blocks else None,
                "total_pages": "",
                "low_text_pages": "",
            })
            continue

        info = extract_text_and_blocks(pdf, out_dir, args.also_blocks)
        summary.append(info)

        if info.get("status") != "OK":
            print(f"[SKIP] {pdf.name} | {info.get('status')} | {info.get('error','')}")
            continue

        print(f"[OK] {pdf.name} -> {Path(info['txt']).name} | low_text_pages={info['low_text_pages']}/{info['total_pages']}")

    # まとめCSV（運用が楽になる）
    summary_csv = out_dir / "_extract_summary.csv"
    fieldnames = ["pdf", "status", "error", "txt", "blocks", "total_pages", "low_text_pages"]

    if args.append_summary:
        file_exists = summary_csv.exists() and summary_csv.stat().st_size > 0
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerows(summary)
    else:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary)

    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()