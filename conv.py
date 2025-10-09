import pypandoc
import os
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import fitz  # PyMuPDF library
import traceback # NEW: Import traceback for detailed error reporting

# --- CONFIGURATION ---
EPUB_SOURCE_DIR = Path("epubs")
PDF_SOURCE_DIR = Path("pdfs")
OUTPUT_DIR = Path("converted")


def extract_clean_text_from_pdf(
    pdf_path: Path, 
    header_margin: float = 0.10, 
    footer_margin: float = 0.10
) -> str:
    """
    Extracts clean text from a PDF file, intelligently removing headers and footers,
    with detailed debug output.
    """
    full_markdown_text = ""
    
    try:
        # Use a 'with' statement for safe file handling
        with fitz.open(pdf_path) as doc:
            # --- NEW: Get page count immediately after opening ---
            num_pages = len(doc)
            print(f"        - Successfully opened. Found {num_pages} pages.")

            # Iterate through each page of the PDF
            for i, page in enumerate(doc, 1):
                print(f"        - Processing Page {i}/{num_pages}...")
                
                # Define the vertical content area for the current page
                page_height = page.rect.height
                header_threshold = page_height * header_margin
                footer_threshold = page_height * (1 - footer_margin)

                # Get all text blocks with their coordinates
                blocks = page.get_text("blocks")
                
                # Filter out blocks that are likely headers or footers
                content_blocks = []
                for block in blocks:
                    block_y0, block_y1 = block[1], block[3]
                    if not (block_y1 < header_threshold or block_y0 > footer_threshold):
                        content_blocks.append(block)
                
                # --- NEW: Detailed block-level logging ---
                print(f"          - Found {len(blocks):>3} total text blocks.")
                print(f"          - Kept  {len(content_blocks):>3} content blocks (removed {len(blocks) - len(content_blocks)}).")
                
                # Sort the content blocks by vertical position for correct reading order
                content_blocks.sort(key=lambda b: b[1])

                # Join the text from the content blocks
                page_text = " ".join([block[4].replace('\n', ' ') for block in content_blocks])
                full_markdown_text += page_text + "\n\n"

        # --- FIXED: This print statement now uses the saved page count ---
        print(f"      -> Finished cleaning. Extracted text from {num_pages} pages.")
        return full_markdown_text

    except Exception as e:
        # --- NEW: Re-raise the exception to be caught by the main loop ---
        # This allows us to provide more context in the main error handler.
        print(f"      -> An error occurred inside PDF extractor: {e}")
        raise


def convert_and_clean_file(source_path: Path, output_path: Path):
    """
    Converts a source file (EPUB or PDF) to clean Markdown.
    """
    file_suffix = source_path.suffix.lower()

    if file_suffix == '.epub':
        # (EPUB logic remains unchanged)
        print("      -> Using EPUB processing pipeline...")
        html_content = pypandoc.convert_file(str(source_path), to='html')
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(["style", "script", "link"]):
            tag.extract()
        for a_tag in soup.find_all('a'):
            a_tag.unwrap()
        for tag in soup.find_all(True):
            if 'style' in tag.attrs:
                del tag.attrs['style']
        markdown_text = md(str(soup), heading_style="ATX")

    elif file_suffix == '.pdf':
        print("      -> Using PDF processing pipeline (PyMuPDF)...")
        markdown_text = extract_clean_text_from_pdf(source_path)

    else:
        raise ValueError(f"Unsupported file type: '{file_suffix}'")

    output_path.write_text(markdown_text, encoding='utf-8')


def batch_convert_files(source_dir: Path, output_dir: Path, file_pattern: str):
    """
    Main batch processing loop with enhanced error reporting.
    """
    if not source_dir.is_dir():
        print(f"⚠️  Warning: Source directory not found at '{source_dir}'. Skipping.")
        return

    print("\n" + "=" * 50)
    print(f"Processing files in '{source_dir.resolve()}'")
    print(f"Looking for files matching: '{file_pattern}'")
    print("=" * 50)

    source_files = list(source_dir.rglob(file_pattern))
    if not source_files:
        print("No files found to process. Nothing to do.")
        return

    total_files = len(source_files)
    print(f"✅ Found {total_files} file(s) to process.")
    
    success_count, failure_count, skipped_count = 0, 0, 0

    for i, source_path in enumerate(source_files, 1):
        print(f"\n[{i}/{total_files}] Processing: '{source_path}'")

        try:
            relative_path = source_path.relative_to(source_dir)
            output_path = output_dir.joinpath(relative_path).with_suffix(".md")

            if output_path.exists():
                print(f"   ⏭️  SKIPPING: Output file already exists at '{output_path}'")
                skipped_count += 1
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   Converting to: '{output_path}'")
            convert_and_clean_file(source_path, output_path)

            print(f"   ✅ Successfully converted.")
            success_count += 1

        except Exception as e:
            # --- NEW: Enhanced, detailed error reporting ---
            print(f"   ❌ FATAL ERROR: An unexpected error occurred with '{source_path}'.")
            print(f"      Error Type: {type(e).__name__}, Message: {e}")
            print("      " + "-"*20 + " TRACEBACK " + "-"*20)
            # This will print the exact line numbers and function calls that led to the error.
            traceback.print_exc()
            print("      " + "-"*51)
            failure_count += 1

    print("\n" + "-" * 30)
    print(f"Batch summary for '{source_dir}':")
    print(f"   - Success: {success_count}")
    print(f"   - Failures: {failure_count}")
    print(f"   - Skipped: {skipped_count}")
    print("-" * 30)


def main():
    print("Starting batch conversion process...")
    batch_convert_files(EPUB_SOURCE_DIR, OUTPUT_DIR, "*.epub")
    batch_convert_files(PDF_SOURCE_DIR, OUTPUT_DIR, "*.pdf")
    print("\nAll processing complete.")


if __name__ == "__main__":
    main()