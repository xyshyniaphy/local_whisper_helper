import os
import re
import pypandoc
import fitz  # PyMuPDF
from pathlib import Path
import traceback
from collections import defaultdict

# --- CONFIGURATION ---
# Define a SINGLE source directory for all input files.
# The script will recursively search for files within this directory.
SOURCE_DIR = Path("source_files")

# Define the single output directory for all converted and split files.
# The original folder structure from SOURCE_DIR will be replicated here.
OUTPUT_DIR = Path("converted_markdown")

# --- 1. HELPER AND CONVERSION FUNCTIONS ---

def _clean_and_remove_all_whitespace(text: str) -> str:
    """
    Aggressively cleans text by removing all whitespace from each line.
    - Splits text into lines.
    - For each line, removes ALL whitespace characters (spaces, tabs, Unicode spaces).
    - Discards any line that becomes empty after cleaning (e.g., blank lines, page numbers).
    - Joins the cleaned, non-empty lines back together.
    """
    cleaned_lines = []
    for line in text.split('\n'):
        # Use regex to remove all whitespace characters (\s matches spaces, tabs, newlines, etc.)
        # This also handles Chinese full-width spaces.
        cleaned_line = re.sub(r'\s+', '', line)
        
        # Only keep lines that still have content after cleaning.
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
            
    return '\n'.join(cleaned_lines)

def convert_with_pandoc(source_path: Path) -> str:
    """
    Converts DOCX, DOC, and EPUB files to clean Markdown using pypandoc.
    """
    file_type = source_path.suffix.upper()[1:]
    print(f"      -> Using pypandoc for {file_type} conversion...")
    extra_args = ['--no-highlight', '--strip-comments', '--extract-media=""', '--wrap=none']
    markdown_string = pypandoc.convert_file(str(source_path), to='gfm', extra_args=extra_args)
    cleaned_markdown = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_string)
    print(f"      -> {file_type} converted.")
    return cleaned_markdown

def convert_pdf_with_pymupdf(source_path: Path) -> str:
    """
    Extracts raw text from a PDF file using PyMuPDF.
    """
    print("      -> Using PyMuPDF for PDF text extraction...")
    full_text = ""
    try:
        with fitz.open(source_path) as doc:
            num_pages = len(doc)
            for page in doc:
                full_text += page.get_text() + "\n"
        
        cleaned_text = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', '', full_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'www\.[^\s/$.?#].[^\s]*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\S+@\S+\.\S+', '', cleaned_text)

        print(f"      -> PDF text extracted from {num_pages} pages.")
        return cleaned_text
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
            # The line is already stripped of all whitespace by the cleaning function.
            # We just need to remove the trailing newline character for matching.
            clean_line = line.strip()
            match = lesson_pattern.match(clean_line)
            
            if match:
                # The matched line is already clean and can be used as the key.
                safe_key = re.sub(r'[\\/*?:"<>|]', "", clean_line)
                current_lesson_key = safe_key
            
            # Append the original line (with its newline) to the content list.
            lessons[current_lesson_key].append(line)
        
        if len(lessons) <= 1 and "_introduction" in lessons:
            print("   -> No lesson markers found. No aggregation performed.")
            return

        print(f"   -> Found {len(lessons)} unique lessons/parts. Saving aggregated files...")
        base_name = md_file_path.stem
        output_dir = md_file_path.parent

        for key, content_lines in lessons.items():
            content = "".join(content_lines)
            
            # Don't save if the final content is empty.
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

def convert_file(source_path: Path, output_path: Path) -> bool:
    """
    Dispatcher function that routes a file to the correct conversion utility,
    then applies the aggressive whitespace cleaning before saving.
    """
    file_suffix = source_path.suffix.lower()
    markdown_content = ""

    if file_suffix in ['.doc', '.docx', '.epub']:
        markdown_content = convert_with_pandoc(source_path)
    elif file_suffix == '.pdf':
        markdown_content = convert_pdf_with_pymupdf(source_path)
    else:
        print(f"      -> Unsupported file type: '{file_suffix}'. Skipping.")
        return False

    # Apply the aggressive whitespace removal.
    final_content = _clean_and_remove_all_whitespace(markdown_content)
    
    output_path.write_text(final_content, encoding='utf-8')
    return True

def batch_process_directory(source_dir: Path, output_dir: Path):
    """
    Main batch processing loop. It finds all supported files in a source
    directory, converts them, and then attempts to aggregate them.
    """
    if not source_dir.is_dir():
        print(f"⚠️  Error: Source directory not found at '{source_dir}'. Please create it and add files.")
        return

    print("\n" + "=" * 60)
    print(f"Processing files in: '{source_dir.resolve()}'")
    print("=" * 60)

    supported_patterns = ["*.doc", "*.docx", "*.pdf", "*.epub"]
    source_files = [p for pattern in supported_patterns for p in source_dir.rglob(pattern)]

    if not source_files:
        print("No supported files found to process.")