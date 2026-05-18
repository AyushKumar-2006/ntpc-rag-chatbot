# scripts/pdf_extractor.py

import json
import os
from pathlib import Path

import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_FILE = RAW_DIR / "pdf_texts.json"


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                text_parts.append(
                    f"\n\n--- Page {page_number} of {total_pages} ---\n{page_text}"
                )

    return "\n".join(text_parts)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    results = []

    print(f"Found {len(pdf_files)} PDF files.")
    print(f"Reading from: {PDF_DIR}")

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\nExtracting {index}/{len(pdf_files)}: {pdf_path.name}")

        try:
            text = extract_text_from_pdf(pdf_path)

            if not text.strip():
                print(f"Warning: No text extracted from {pdf_path.name}")
                continue

            results.append(
                {
                    "title": pdf_path.stem,
                    "file": pdf_path.name,
                    "url": f"/reports/{pdf_path.name}",
                    "text": text,
                }
            )

            print(f"Extracted {len(text)} characters")

        except Exception as e:
            print(f"Failed to extract {pdf_path.name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(results)} PDFs extracted.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()