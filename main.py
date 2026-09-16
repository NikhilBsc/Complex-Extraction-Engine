"""
main.py
-------
PDF Business-Field Extraction Engine — End-to-End Pipeline (Phase 10)

Usage
-----
    # Basic (uses documents/ folder auto-discovery)
    python main.py

    # Specific PDF
    python main.py path/to/document.pdf

    # With options
    python main.py path/to/document.pdf --ocr --output-dir results/

    # Skip GPT (dry run: ingestion + preprocessing + masking + segmentation only)
    python main.py path/to/document.pdf --dry-run

    # Include flagged fields in output
    python main.py path/to/document.pdf --include-flagged

Pipeline
--------
  Phase 1-2  : Document Ingestion     (native text + OCR fallback)
  Phase 3    : Preprocessing          (clean + structural normalisation)
  Phase 4    : Selective Masking      (Presidio PII detection, local only)
  Phase 5    : Segmentation           (token-budget-aware logical chunks)
  Phase 6    : GPT Extraction         (dynamic field discovery, gpt-4o)
  Phase 7    : Validation             (grounding + quality checks)
  Phase 8    : Consolidation          (deduplication + conflict resolution)
  Phase 9    : Excel Output           (3-sheet workbook)
"""

from __future__ import annotations

import argparse
import logging
import sys
import io
import time
from datetime import datetime
from pathlib import Path

# UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)-30s | %(message)s"
    datefmt = "%H:%M:%S"

    # Console — INFO and above only (clean output)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING if not verbose else logging.DEBUG)
    console.setFormatter(logging.Formatter("  %(levelname)-8s %(message)s", datefmt=datefmt))

    # File — everything
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    return log_file


logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Progress printer
# ─────────────────────────────────────────────────────────────────────────────

class _Phase:
    def __init__(self, name: str, number: int):
        self.name = name
        self.number = number
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        print(f"\n  {'─'*56}")
        print(f"  Phase {self.number}  {self.name}")
        print(f"  {'─'*56}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type is None:
            print(f"  ✓  Done  ({elapsed:.1f}s)")
        return False   # don't suppress exceptions


def _banner(pdf_path: Path) -> None:
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   PDF Business-Field Extraction Engine  v1.0        ║")
    print("  ║   Accuracy Target: ≥95%  |  GPT-4o  |  Local PII   ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print(f"\n  Document : {pdf_path.name}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def _print_field_table(fields: list, show_flagged: bool = True) -> None:
    """Print a clean field | value table to console."""
    accepted   = [f for f in fields if f.validation_status == "accepted"]
    flagged    = [f for f in fields if f.validation_status == "flagged"]
    to_display = accepted + (flagged if show_flagged else [])

    if not to_display:
        print("\n  [!] No fields to display.")
        return

    col1 = max(len(f.field_name) for f in to_display)
    col1 = max(col1, 15)
    col2 = 55
    sep  = f"  {'─'*col1}  {'─'*col2}  {'─'*6}  {'─'*5}"

    print(f"\n  {'FIELD':<{col1}}  {'VALUE':<{col2}}  {'CONF':>6}  {'PAGES'}")
    print(sep)

    for f in to_display:
        name  = f.field_name[:col1]
        val   = str(f.value)[:col2]
        conf  = f"{f.confidence:.0%}"
        pages = ",".join(str(p) for p in f.source_pages[:3])
        flag  = " ⚑" if f.validation_status == "flagged" else ""
        conf_mark = " ▲" if f.confidence >= 0.90 else " ●" if f.confidence >= 0.75 else " ▼"
        print(f"  {name:<{col1}}  {val:<{col2}}  {conf:>5}{conf_mark}  p{pages}{flag}")

    print(sep)
    print(f"\n  Total: {len(accepted)} accepted  |  {len(flagged)} flagged")
    if flagged and not show_flagged:
        print(f"  (run with --include-flagged to see {len(flagged)} flagged fields)")


# ─────────────────────────────────────────────────────────────────────────────
# PDF auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_pdf(given_path: str | None) -> Path:
    if given_path:
        p = Path(given_path)
        if not p.exists():
            print(f"  [ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p

    # Auto-discover from documents/
    docs_dir = ROOT / "documents"
    if docs_dir.exists():
        pdfs = sorted(docs_dir.glob("*.pdf")) + sorted(docs_dir.glob("*.PDF"))
        pdfs += sorted(docs_dir.glob("*.Pdf"))
        if pdfs:
            print(f"  [Auto] Found {len(pdfs)} PDF(s) in documents/")
            for i, p in enumerate(pdfs[:5]):
                print(f"    [{i+1}] {p.name}")
            if len(pdfs) == 1:
                return pdfs[0]
            # Use first by default
            print(f"  [Auto] Using: {pdfs[0].name}  (pass path to override)")
            return pdfs[0]

    print("  [ERROR] No PDF path given and no PDFs found in documents/", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: Path,
    run_ocr: bool = False,
    dry_run: bool = False,
    include_flagged: bool = True,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> dict:
    """
    Execute the full extraction pipeline on a single PDF.

    Parameters
    ----------
    pdf_path : Path
    run_ocr : bool
        Force OCR on all pages (slow but handles scanned PDFs).
    dry_run : bool
        Skip GPT phase — useful for testing ingestion/masking pipeline.
    include_flagged : bool
        Include flagged fields in Excel output.
    output_dir : Path | None
        Where to write Excel. Defaults to <project_root>/output/
    verbose : bool
        Enable DEBUG logging.

    Returns
    -------
    dict with keys: fields, report, excel_path, summary
    """
    t_start = time.time()

    # ── Phase 1-2: Ingestion ─────────────────────────────────────────────
    with _Phase("Document Ingestion  (native text + OCR fallback)", 2):
        from ingestion import read_pdf
        pages = read_pdf(pdf_path, run_ocr=run_ocr)
        total_pages = len(pages)
        native_count = sum(1 for p in pages if p.source_type == "native")
        ocr_count    = sum(1 for p in pages if p.source_type == "ocr")
        print(f"  Pages: {total_pages}  |  native={native_count}  ocr={ocr_count}")

    # ── Phase 3: Preprocessing ──────────────────────────────────────────
    with _Phase("Preprocessing  (clean + structural normalisation)", 3):
        from preprocessing import preprocess
        preprocessed = preprocess(pages)
        kv_count  = sum(1 for p in preprocessed if p.structured.has_key_value)
        tbl_count = sum(1 for p in preprocessed if p.structured.has_table)
        print(f"  Pages with KV={kv_count}  |  tables={tbl_count}")

    # ── Phase 4: Selective Masking ───────────────────────────────────────
    with _Phase("Selective Masking  (local Presidio PII detection)", 4):
        from masking.engine import mask_document
        masked_pages, mask_ctx = mask_document(
            preprocessed,
            doc_id=pdf_path.stem,
            save_mask_file=True,
        )
        masked_count = mask_ctx.total_masked()
        print(f"  Entities masked: {masked_count}  |  {mask_ctx.summary()}")

    # ── Phase 5: Segmentation ────────────────────────────────────────────
    with _Phase("Segmentation  (token-budget-aware logical chunks)", 5):
        from extraction.segmenter import segment_document
        segments = segment_document(masked_pages, preprocessed)
        total_tokens = sum(s.estimated_tokens for s in segments)
        kv_segs   = sum(1 for s in segments if s.is_key_value_block)
        tbl_segs  = sum(1 for s in segments if s.is_table)
        print(f"  Segments: {len(segments)}  |  KV={kv_segs}  table={tbl_segs}  ~tokens={total_tokens:,}")

    if dry_run:
        print("\n  [Dry-run] Stopping before GPT phase.")
        print(f"  Pipeline ready — {len(segments)} segments would be sent to GPT.")
        return {
            "fields": [], "report": None,
            "excel_path": None, "summary": None,
            "segments": segments,
        }

    # ── Phase 6: GPT Extraction ──────────────────────────────────────────
    with _Phase("GPT Extraction  (gpt-4o dynamic field discovery)", 6):
        from gpt.extractor import extract_document
        results, gpt_summary = extract_document(segments)
        print(f"  Segments processed : {gpt_summary.segments_processed}")
        print(f"  Fields extracted   : {gpt_summary.total_fields_extracted}")
        print(f"  Total tokens       : {gpt_summary.total_tokens:,}")
        print(f"  Est. cost          : ${gpt_summary.estimated_cost_usd():.4f}")
        print(f"  Fallback calls     : {gpt_summary.segments_with_fallback}")
        if gpt_summary.errors:
            print(f"  [WARN] Errors on {len(gpt_summary.errors)} segment(s)")

    # ── Phase 7: Validation ──────────────────────────────────────────────
    with _Phase("Validation  (grounding + quality checks)", 7):
        from validation.validator import validate_document, build_document_context
        doc_context = build_document_context(masked_pages, preprocessed, segments)
        all_fields, report = validate_document(results, doc_context, mask_ctx)
        print(f"  Total fields  : {report.total_fields}")
        print(f"  Accepted      : {report.accepted}")
        print(f"  Flagged       : {report.flagged}")
        print(f"  Rejected      : {report.rejected}")
        print(f"  Precision est : {report.precision_estimate:.1%}")

    # ── Phase 8: Consolidation ───────────────────────────────────────────
    with _Phase("Consolidation  (deduplication + conflict resolution)", 8):
        from output.consolidator import consolidate, consolidation_summary
        consolidated = consolidate(all_fields, include_flagged=include_flagged)
        summary = consolidation_summary(consolidated)
        print(f"  Unique fields  : {summary['total_unique_fields']}")
        print(f"  Conflicts      : {summary['fields_with_conflicts']}")
        print(f"  Avg confidence : {summary['avg_confidence']:.1%}")

    # ── Phase 9: Excel Output ────────────────────────────────────────────
    with _Phase("Excel Output  (3-sheet workbook)", 9):
        from output.excel_writer import write_excel

        out_dir = output_dir or (ROOT / "output")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = out_dir / f"{pdf_path.stem}_extraction_{timestamp}.xlsx"

        excel_path = write_excel(
            consolidated_fields=consolidated,
            report=report,
            output_path=excel_path,
            doc_name=pdf_path.name,
            extraction_meta={
                "model_used":          ", ".join(gpt_summary.models_used),
                "total_tokens":        gpt_summary.total_tokens,
                "cost_usd":            gpt_summary.estimated_cost_usd(),
                "segments_processed":  gpt_summary.segments_processed,
                "segments_skipped":    gpt_summary.segments_skipped,
            },
        )
        print(f"  Saved to: {excel_path}")

    # ── Console summary ──────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n  {'═'*58}")
    print(f"  EXTRACTION COMPLETE  ({elapsed:.1f}s total)")
    print(f"  {'═'*58}")

    _print_field_table(consolidated, show_flagged=include_flagged)

    print(f"\n  Excel report : {excel_path}")
    print(f"  Log file     : see logs/ directory")
    print()

    return {
        "fields":     consolidated,
        "report":     report,
        "excel_path": excel_path,
        "summary":    summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="PDF Business-Field Extraction Engine — GPT-4o powered, ≥95% accuracy target",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="Path to the PDF to process. If omitted, auto-discovers from documents/",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        default=False,
        help="Force OCR on all pages (slower, for scanned PDFs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run pipeline up to segmentation (no GPT calls)",
    )
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        default=True,
        help="Include flagged fields in output (default: True)",
    )
    parser.add_argument(
        "--no-flagged",
        action="store_true",
        default=False,
        help="Exclude flagged fields from output",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the Excel file (default: output/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging to console",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    log_file = _setup_logging(verbose=args.verbose)

    pdf_path   = _find_pdf(args.pdf)
    output_dir = Path(args.output_dir) if args.output_dir else None
    include_flagged = args.include_flagged and not args.no_flagged

    _banner(pdf_path)
    print(f"  Log file : {log_file}\n")

    try:
        run_pipeline(
            pdf_path=pdf_path,
            run_ocr=args.ocr,
            dry_run=args.dry_run,
            include_flagged=include_flagged,
            output_dir=output_dir,
            verbose=args.verbose,
        )
    except EnvironmentError as e:
        # Missing API key
        print(f"\n  [ERROR] {e}", file=sys.stderr)
        print(f"\n  Add your OpenAI API key to .env:\n    OPENAI_API_KEY=sk-...\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  [Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        logger.exception("Pipeline failed with unhandled exception")
        print(f"\n  [FATAL] {type(e).__name__}: {e}", file=sys.stderr)
        print(f"  Full traceback in log file: {log_file}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
