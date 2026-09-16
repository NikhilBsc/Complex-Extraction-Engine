"""
web/app.py
----------
FastAPI Web Interface for PDF Extraction Engine.
Supports PDF Drag-and-Drop, Real-time Streaming (SSE) for pipeline progress and live field discovery,
and direct Excel report downloading.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion import read_pdf
from preprocessing import preprocess
from masking.engine import mask_document
from extraction.segmenter import segment_document
from gpt.extractor import extract_segment, ExtractionSummary
from validation.validator import validate_document, build_document_context
from output.consolidator import consolidate, consolidation_summary
from output.excel_writer import write_excel

logger = logging.getLogger("web_app")

app = FastAPI(title="PDF Business-Field Extraction Engine UI")

DOCS_DIR = ROOT / "documents"
OUTPUT_DIR = ROOT / "output"
DOCS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

STATIC_DIR = ROOT / "web" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found")
    return index_path.read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{Path(file.filename).stem}_{timestamp}.pdf"
    file_path = DOCS_DIR / safe_filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "status": "success",
        "doc_id": safe_filename,
        "original_name": file.filename,
        "path": str(file_path)
    }


@app.get("/api/stream/{doc_id}")
async def stream_pipeline(doc_id: str):
    pdf_path = DOCS_DIR / doc_id
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Document file not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        def send_event(event_type: str, data: dict):
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        try:
            # --- Step 1: Ingestion ---
            yield send_event("status", {"phase": 1, "step": "ingestion", "msg": "Ingesting document text & OCR evaluation..."})
            await asyncio.sleep(0.1)
            
            pages = read_pdf(pdf_path, run_ocr=True)
            total_pages = len(pages)
            yield send_event("status", {
                "phase": 1, 
                "step": "ingestion_done", 
                "msg": f"Ingested {total_pages} page(s)",
                "total_pages": total_pages
            })

            # --- Step 2: Preprocessing ---
            yield send_event("status", {"phase": 2, "step": "preprocessing", "msg": "Cleaning text & normalizing layout..."})
            preprocessed = preprocess(pages)
            kv_count = sum(1 for p in preprocessed if p.structured.has_key_value)
            yield send_event("status", {"phase": 2, "step": "preprocessing_done", "msg": f"Layout normalized ({kv_count} KV sections found)"})

            # --- Step 3: PII Masking ---
            yield send_event("status", {"phase": 3, "step": "masking", "msg": "Applying selective PII masking..."})
            masked_pages, mask_ctx = mask_document(preprocessed, doc_id=pdf_path.stem, save_mask_file=True)
            yield send_event("status", {"phase": 3, "step": "masking_done", "msg": f"PII Masked: {mask_ctx.total_masked()} sensitive token(s)"})

            # --- Step 4: Segmentation ---
            yield send_event("status", {"phase": 4, "step": "segmentation", "msg": "Segmenting text into token-budget chunks..."})
            segments = segment_document(masked_pages, preprocessed)
            total_segs = len(segments)
            yield send_event("status", {
                "phase": 4, 
                "step": "segmentation_done", 
                "msg": f"Created {total_segs} logical segment(s)",
                "total_segments": total_segs
            })

            # --- Step 5: AI Field Extraction ---
            yield send_event("status", {"phase": 5, "step": "extraction_start", "msg": "Starting AI Field Discovery...", "total_segments": total_segs})
            
            summary = ExtractionSummary(total_segments=total_segs)
            raw_results = []
            
            for idx, seg in enumerate(segments, start=1):
                yield send_event("extraction_progress", {
                    "current": idx,
                    "total": total_segs,
                    "segment_id": seg.segment_id,
                    "pages": seg.page_numbers,
                    "msg": f"Processing segment {idx}/{total_segs} (pages {seg.page_numbers})"
                })
                
                # Execute single segment extraction (run in thread pool to prevent blocking event loop)
                result = await asyncio.to_thread(extract_segment, seg)
                
                if result is not None:
                    summary.add_result(result)
                    raw_results.append(result)
                    
                    # Stream discovered fields live to UI
                    fields_data = [
                        {
                            "field_name": f.field_name,
                            "value": f.value,
                            "confidence": f.confidence,
                            "evidence": f.evidence,
                            "segment_id": f.segment_id,
                            "page_numbers": f.page_numbers
                        } for f in result.fields
                    ]
                    yield send_event("fields_batch", {
                        "segment_id": seg.segment_id,
                        "fields": fields_data
                    })
                
                # Rate limit sleep
                await asyncio.sleep(0.5)

            yield send_event("status", {
                "phase": 5, 
                "step": "extraction_done", 
                "msg": f"Extraction complete ({summary.total_fields_extracted} raw fields discovered)"
            })

            # --- Step 6: Validation & Grounding ---
            yield send_event("status", {"phase": 6, "step": "validation", "msg": "Validating grounding & rehydrating values..."})
            doc_context = build_document_context(masked_pages, preprocessed, segments)
            all_fields, report = validate_document(raw_results, doc_context, mask_ctx)
            
            yield send_event("status", {
                "phase": 6, 
                "step": "validation_done", 
                "msg": f"Validation: {report.accepted} accepted, {report.flagged} flagged, {report.rejected} rejected"
            })

            # --- Step 7: Consolidation & Excel Export ---
            yield send_event("status", {"phase": 7, "step": "consolidation", "msg": "Consolidating fields & building Excel report..."})
            consolidated = consolidate(all_fields, include_flagged=True)
            
            excel_filename = f"{pdf_path.stem}_report.xlsx"
            excel_path = OUTPUT_DIR / excel_filename
            
            write_excel(
                consolidated_fields=consolidated,
                report=report,
                output_path=excel_path,
                doc_name=pdf_path.name,
                extraction_meta={
                    "model_used": ", ".join(summary.models_used),
                    "total_tokens": summary.total_tokens,
                    "cost_usd": summary.estimated_cost_usd(),
                    "segments_processed": summary.segments_processed,
                    "segments_skipped": summary.segments_skipped,
                }
            )

            # Final payload sent to UI
            final_fields = [
                {
                    "field_name": cf.field_name,
                    "value": cf.value,
                    "confidence": cf.confidence,
                    "source_pages": cf.source_pages,
                    "occurrence_count": cf.occurrence_count,
                    "has_conflict": cf.has_conflict,
                    "all_values": cf.all_values,
                    "validation_status": cf.validation_status,
                    "best_evidence": cf.best_evidence
                } for cf in consolidated
            ]

            yield send_event("complete", {
                "excel_download_url": f"/api/download/{excel_filename}",
                "total_fields": len(consolidated),
                "precision": report.precision_estimate,
                "summary": consolidation_summary(consolidated),
                "final_fields": final_fields
            })

        except Exception as exc:
            logger.exception("Streaming pipeline error")
            yield send_event("error", {"msg": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Requested file not found")
    return FileResponse(
        file_path, 
        filename=filename, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8050)
