"""PDF loading with an OCR fallback.

Most resumes in data/resume are scans (a JPEG wrapped in a PDF), so they have no
text layer. For every page we first try the normal text layer; if a page has
(almost) no text we render it to an image and run OCR on it.

OCR results are cached on disk (one small JSON per PDF), so each scanned resume
is only OCR'd once.
"""
import hashlib
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document

MIN_TEXT_CHARS = 30      # a page with fewer characters than this is treated as a scan
TARGET_WIDTH_PX = 1600   # render width used for OCR
TILE_HEIGHT_PX = 1800    # very tall pages are OCR'd in vertical slices
TILE_OVERLAP_PX = 100    # overlap between slices so lines are not cut in half

_engine = None


def _get_engine():
    """Create the OCR engine once, on first use (loading the models takes a moment)."""
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR
        _engine = RapidOCR()
    return _engine


def file_hash(path: Path) -> str:
    """MD5 of the file contents; used to detect duplicate files and to name cache files."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ocr_page(page) -> str:
    """Render one PDF page to an image and return the OCR'd text."""
    import pymupdf

    width = page.rect.width or 1
    zoom = max(1.0, min(TARGET_WIDTH_PX / width, 4.0))
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False
    )
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR, which the OCR engine expects

    engine = _get_engine()
    lines: List[str] = []
    y = 0
    while y < img.shape[0]:
        result = engine(img[y:y + TILE_HEIGHT_PX])
        txts = getattr(result, "txts", None)
        if txts:
            recent = set(lines[-5:])  # drop lines repeated because of the tile overlap
            lines.extend(t for t in txts if t not in recent)
        if y + TILE_HEIGHT_PX >= img.shape[0]:
            break
        y += TILE_HEIGHT_PX - TILE_OVERLAP_PX
    return "\n".join(lines)


def load_pdf(
    pdf_path,
    cache_dir: Optional[Path] = None,
    ocr: bool = True,
    digest: Optional[str] = None,
) -> List[Document]:
    """Load a PDF as one Document per non-empty page, OCR-ing pages that have no text layer."""
    import pymupdf

    pdf_path = Path(pdf_path)
    digest = digest or file_hash(pdf_path)

    cache: dict = {}
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{pdf_path.stem}_{digest[:8]}.json"
        if cache_file.exists():
            cache = json.loads(cache_file.read_text(encoding="utf-8"))

    docs: List[Document] = []
    cache_changed = False
    with pymupdf.open(pdf_path) as pdf:
        total_pages = len(pdf)
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            used_ocr = False
            if ocr and len(text) < MIN_TEXT_CHARS:
                used_ocr = True
                key = str(i)
                if key in cache:
                    text = cache[key]
                else:
                    try:
                        text = _ocr_page(page)
                        cache[key] = text
                        cache_changed = True
                    except Exception as e:  # e.g. rapidocr not installed
                        print(f"[ERROR] OCR failed for {pdf_path.name} page {i}: {e}")
                        text = ""
            if not text.strip():
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "source_file": pdf_path.name,
                    "candidate_id": pdf_path.stem,
                    "file_type": "pdf",
                    "file_hash": digest,
                    "page": i,
                    "total_pages": total_pages,
                    "ocr": used_ocr,
                },
            ))

    if cache_changed and cache_file is not None:
        cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return docs