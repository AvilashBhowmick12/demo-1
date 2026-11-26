from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from src.pdf_ocr import load_pdf, file_hash

def load_all_documents(data_dir: str) -> List[Any]: 
    """
    Load all supported files from the data directory and convert to LangChain document structure 
    Supported: PDF (with OCR for scanned pages), TXT, CSV, Excel, Word
    """
    # Use project root data folder 
    data_path = Path(data_dir).resolve() 
    print(f"[DEBUG] Data path: {data_path}") 
    documents = [] 
    
    # PDF files: text layer first, OCR fallback for scanned pages (results cached)
    pdf_files = sorted(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")
    cache_dir = data_path / "ocr_cache"
    seen_hashes = set()
    duplicates = 0
    ocr_pages = 0
    text_pdfs = 0       # PDFs whose pages all have a real text layer: no OCR needed
    scanned_pdfs = 0    # PDFs where at least one page had to be OCR'd
    for n, pdf_file in enumerate(pdf_files, 1):
        try:
            digest = file_hash(pdf_file)
            if digest in seen_hashes:
                duplicates += 1
                print(f"[DEBUG] Skipping duplicate file: {pdf_file.name}")
                continue
            seen_hashes.add(digest)
            loaded = load_pdf(pdf_file, cache_dir=cache_dir, digest=digest)
            pages_ocr = sum(1 for d in loaded if d.metadata.get("ocr"))
            ocr_pages += pages_ocr
            if pages_ocr:
                scanned_pdfs += 1
            else:
                text_pdfs += 1
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load pdf {pdf_file}: {e}")
        if n % 25 == 0:
            print(f"[DEBUG] PDFs processed: {n}/{len(pdf_files)}")
    print(f"[DEBUG] PDFs loaded: {len(seen_hashes)} unique, {duplicates} duplicates skipped")
    print(f"[DEBUG]   {text_pdfs} PDFs had a normal text layer (read directly, no OCR); "
          f"{scanned_pdfs} were scanned ({ocr_pages} pages read with OCR)")

    # Text files
    text_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(text_files)} Text files")
    for text_file in text_files:
        try:
            loaded = load_single_document(text_file)
            print(f"[DEBUG] Loaded {len(loaded)} Text docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Text file {text_file}: {e}")


    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(
                file_path=str(csv_file),
                encoding="utf-8"
            )
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV file {csv_file}: {e}")

        # Excel files
    excel_files = list(data_path.glob('**/*.xlsx')) + list(data_path.glob('**/*.xls'))
    print(f"[DEBUG] Found {len(excel_files)} Excel files")
    for excel_file in excel_files:
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {excel_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel file {excel_file}: {e}")

    # Word files
    word_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(word_files)} Word files")
    pdf_stems = {(p.parent, p.stem.lower()) for p in pdf_files}
    for word_file in word_files:
        if (word_file.parent, word_file.stem.lower()) in pdf_stems:
            print(f"[DEBUG] Skipping {word_file.name}: a PDF with the same name exists (same resume)")
            continue
        try:
            loaded = load_single_document(word_file)   # no OCR: the text is read straight from the file
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {word_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word file {word_file}: {e}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents


def load_single_document(path, cache_dir=None) -> List[Any]:
    """Load one resume (pdf / docx / txt) with the same metadata the bulk loader produces."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path, cache_dir=cache_dir)
    if ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext == ".txt":
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for d in docs:
        d.metadata.update({"source_file": path.name, "candidate_id": path.stem, "file_type": ext.lstrip("."),
                           "page": 0, "ocr": False})
    return docs