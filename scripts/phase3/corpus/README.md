# Phase3 Corpus: PDF -> Text/Blocks extraction

This folder contains utilities to extract *embedded text* from Integrated Report PDFs and save them as machine-readable artifacts for Phase 3 (tagging / evidence extraction).

## Input / Output

- **Input (PDFs):** `data/source/corpus/*.pdf`
- **Output (recommended):** `data/interim/phase3/<run_id>/corpus_text/`
  - `<company>.txt` : page-delimited extracted text
  - `<company>_blocks.csv` : optional block-level text (with bbox) for evidence quoting
  - `_extract_summary.csv` : extraction summary (pages, low-text pages)

> `run_id` can be reused while you incrementally add companies (e.g., test 1 company first, then add more into the same folder).

## Quick start (one company)

```bash
RUN_ID=$(date "+%Y%m%d_%H%M")
mkdir -p data/interim/phase3/${RUN_ID}/corpus_text

python3 scripts/phase3/corpus/extract_pdf_text.py \
  --input "data/source/corpus/キリンホールディングス.pdf" \
  --out_dir "data/interim/phase3/${RUN_ID}/corpus_text" \
  --also_blocks \
  --skip_existing \
  --append_summary
```

## Run for all PDFs in the corpus directory

```bash
RUN_ID=$(date "+%Y%m%d_%H%M")
mkdir -p data/interim/phase3/${RUN_ID}/corpus_text

python3 scripts/phase3/corpus/extract_pdf_text.py \
  --input_dir "data/source/corpus" \
  --out_dir "data/interim/phase3/${RUN_ID}/corpus_text" \
  --also_blocks \
  --skip_existing \
  --append_summary
```

## How to interpret the log

The script prints:

- `low_text_pages = X / total_pages`

If `X` is small (often 0–a few pages), the PDF likely contains embedded text and you can proceed without OCR.
If `X` is large, the PDF may be image/scanned; consider OCR *only for the needed chapters/pages* (do not OCR the whole file).

## Notes

- The extraction assumes **embedded-text PDFs**. If a PDF is mostly scanned images, you will see many `LIKELY_IMAGE_OR_NO_TEXT` pages.
- `--skip_existing` allows reusing the same `run_id` output folder while incrementally processing more companies.
- `--append_summary` appends to `_extract_summary.csv` instead of overwriting it.
