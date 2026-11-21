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
    for n, pdf_file in enumerate(pdf_files, 1):
        try:
            digest = file_hash(pdf_file)
            if digest in seen_hashes:
                duplicates += 1
                print(f"[DEBUG] Skipping duplicate file: {pdf_file.name}")
                continue
            seen_hashes.add(digest)
            loaded = load_pdf(pdf_file, cache_dir=cache_dir, digest=digest)
            ocr_pages += sum(1 for d in loaded if d.metadata.get("ocr"))
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] failed to load pdf {pdf_file}: {e}")
        if n % 25 == 0:
            print(f"[DEBUG] PDFs processed: {n}/{len(pdf_files)}")
    print(f"[DEBUG] PDFs loaded: {len(seen_hashes)} unique, {duplicates} duplicates skipped, "
          f"{ocr_pages} pages came from OCR")

    # Text files
    text_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(text_files)} Text files")
    for text_file in text_files:
        try:
            loader = TextLoader(str(text_file), encoding="utf-8")
            loaded = loader.load()
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
    for word_file in word_files:
        try:
            loader = Docx2txtLoader(str(word_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {word_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word file {word_file}: {e}")

    print(f"[DEBUG] Total documents loaded: {len(documents)}")
    return documents