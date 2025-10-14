import os
import re
import pypandoc
import fitz  # PyMuPDF
from pathlib import Path
import traceback
from collections import defaultdict

# --- CONFIGURATION ---
# Define a SINGLE source directory for all input files.
SOURCE_DIR = Path("source_files")

# Define the single output directory for all converted and split files.
OUTPUT_DIR = Path("converted_markdown")

# --- 1. HELPER, CLEANING, AND CONVERSION FUNCTIONS ---

def _sanitize_filename_part(name: str, max_length: int = 100) -> str:
    """
    Removes illegal characters from a string to make it a valid filename part.
    """
    # Remove illegal characters for Windows/Linux/macOS filenames
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    # Trim leading/trailing whitespace that might have been missed
    sanitized = sanitized.strip()
    # Truncate to a reasonable length
    return sanitized[:max_length]

def _clean_converted_text(raw_text: str) -> str:
    """
    A common function to clean raw text in three stages.
    """
    # Stage 1: Remove links and URLs.
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', raw_text)
    text = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'www\.[^\s/$.?#].[^\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Stage 2: Use regex to remove any remaining HTML tags.
    text = re.sub(r'<[^>]+>', '', text)

    # Stage 3: Process line-by-line for final cleanup.
    text = re.sub(r'[^\S\n]+', '', text) # Remove inline whitespace
    lines = text.split('\n')
    processed_lines = [ '' if line.isdigit() else line for line in lines ]
            
    return '\n'.join(processed_lines)

def convert_with_pandoc(source_path: Path) -> str:
    """
    Converts DOCX, DOC, EPUB, and HTML files to raw Markdown using pypandoc.
    """
    file_type = source_path.suffix.upper()[1:]
    print(f"      -> Using pypandoc for {file_type} conversion...")
    extra_args = ['--no-highlight', '--strip-comments', '--extract-media=""', '--wrap=none']
    markdown_string = pypandoc.convert_file(str(source_path), to='gfm', extra_args=extra_args)
    print(f"      -> {file_type} converted.")
    return markdown_string

def convert_pdf_with_pymupdf(source_path: Path) -> str:
    """
    Extracts raw text from a PDF file, preserving its internal newlines.
    """
    print("      -> Using PyMuPDF for PDF text extraction...")
    full_text = ""
    try:
        with fitz.open(source_path) as doc:
            num_pages = len(doc)
            for page in doc:
                full_text += page.get_text()
        
        print(f"      -> PDF text extracted from {num_pages} pages.")
        return full_text
    except Exception as e:
        print(f"      -> Error during PDF processing: {e}")
        raise

# --- 2. FILE SPLITTING AND AGGREGATION LOGIC ---

def split_and_aggregate_by_lesson(md_file_path: Path):
    """
    Reads a pre-cleaned markdown file and aggregates content by lesson markers.
    """
    print(f"   -> Aggregating content by lesson markers in '{md_file_path.name}'...")
    lesson_pattern = re.compile(r"^(第.*课)$")
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lessons = defaultdict(list)
        current_lesson_key = "_introduction" 

        for line in lines:
            stripped_line = line.strip()
            match = lesson_pattern.match(stripped_line)
            
            if match:
                safe_key = re.sub(r'[\\/*?:"<>|]', "", stripped_line)
                current_lesson_key = safe_key
            else:
                lessons[current_lesson_key].append(line)
        
        if len(lessons) <= 1 and "_introduction" in lessons:
            print("   -> No lesson markers found. No aggregation performed.")
            return

        print(f"   -> Found {len(lessons)} unique lessons/parts. Saving aggregated files...")
        base_name = md_file_path.stem
        output_dir = md_file_path.parent

        for key, content_lines in lessons.items():
            content = "".join(content_lines)
            if not content.strip():
                continue
            new_filename = f"{base_name}_{key}.md"
            output_path = output_dir / new_filename
            output_path.write_text(content, encoding='utf-8')
            print(f"      - Saved aggregated file: '{output_path.name}'")
            
    except Exception as e:
        print(f"   -> ERROR during aggregation of '{md_file_path.name}': {e}")
        traceback.print_exc()

# --- 3. MAIN DISPATCHER AND BATCH PROCESSING ---

def convert_file(source_path: Path, output_path: Path) -> tuple[bool, str]:
    """
    Dispatcher that gets raw text, calls the cleaner, saves the file,
    and returns the success status and final content.
    """
    raw_content = ""
    file_suffix = source_path.suffix.lower()

    if file_suffix in ['.doc', '.docx', '.epub', '.html']:
        raw_content = convert_with_pandoc(source_path)
    elif file_suffix == '.pdf':
        raw_content = convert_pdf_with_pymupdf(source_path)
    else:
        print(f"      -> Unsupported file type: '{file_suffix}'. Skipping.")
        return False, ""

    final_content = _clean_converted_text(raw_content)
    output_path.write_text(final_content, encoding='utf-8')
    return True, final_content

def batch_process_directory(source_dir: Path, output_dir: Path):
    """
    Main batch processing loop with conditional output filename logic.
    """
    if not source_dir.is_dir():
        print(f"⚠️  Error: Source directory not found at '{source_dir}'. Please create it and add files.")
        return

    print("\n" + "=" * 60)
    print(f"Processing files in: '{source_dir.resolve()}'")
    print("=" * 60)

    supported_patterns = ["*.doc", "*.docx", "*.pdf", "*.epub", "*.html"]
    source_files = [p for pattern in supported_patterns for p in source_dir.rglob(pattern)]

    if not source_files:
        print("No supported files found to process.")
        return

    total_files = len(source_files)
    print(f"✅ Found {total_files} file(s) to process.")
    
    success_count, failure_count = 0, 0

    for i, source_path in enumerate(source_files, 1):
        print(f"\n[{i}/{total_files}] Processing: '{source_path}'")

        try:
            relative_path = source_path.relative_to(source_dir)
            initial_output_path = output_dir / relative_path.with_suffix(".md")
            initial_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   -> Converting to: '{initial_output_path}'")
            conversion_success, final_content = convert_file(source_path, initial_output_path)

            if conversion_success:
                final_md_path = initial_output_path

                # NEW: Conditional filename logic
                if source_path.name.startswith("ZB"):
                    # Get the first line of the cleaned content as a prefix
                    first_line = final_content.split('\n', 1)[0] if final_content else ""
                    prefix = _sanitize_filename_part(first_line)

                    if prefix:
                        # Construct the new filename and path
                        new_name = f"{prefix}_{initial_output_path.stem}{initial_output_path.suffix}"
                        new_path = initial_output_path.with_name(new_name)
                        
                        # Rename the file that was just saved
                        initial_output_path.rename(new_path)
                        print(f"   -> Renamed to: '{new_path.name}'")
                        final_md_path = new_path # Use the new path for aggregation

                # Call the aggregation function with the correct final path
                #split_and_aggregate_by_lesson(final_md_path)
                print(f"   ✅ Successfully processed '{source_path.name}'.")
                success_count += 1
            else:
                failure_count += 1

        except Exception as e:
            print(f"   ❌ FATAL ERROR processing '{source_path.name}'.")
            print(f"      Error Type: {type(e).__name__}, Message: {e}")
            traceback.print_exc()
            failure_count += 1

    print("\n" + "-" * 30)
    print(f"Batch summary for '{source_dir}':")
    print(f"   - Success: {success_count}")
    print(f"   - Failures: {failure_count}")
    print("-" * 30)

def main():
    """Main entry point for the script."""
    print("Starting document conversion and aggregation process...")
    batch_process_directory(SOURCE_DIR, OUTPUT_DIR)
    print("\nAll processing complete.")

if __name__ == "__main__":
    main()