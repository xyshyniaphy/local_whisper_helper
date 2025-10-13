import os
import re
import pypandoc
import fitz  # PyMuPDF
from pathlib import Path
import traceback

# --- CONFIGURATION ---
# Define a SINGLE source directory for all input files.
# The script will recursively search for files within this directory.
SOURCE_DIR = Path("source_files")

# Define the single output directory for all converted and split files.
# The original folder structure from SOURCE_DIR will be replicated here.
OUTPUT_DIR = Path("converted_markdown")

# --- 1. HELPER AND CONVERSION FUNCTIONS ---

def _clean_and_normalize_whitespace(text: str) -> str:
    """
    Normalizes and cleans whitespace from the converted text.
    - Replaces Chinese full-width spaces with standard spaces.
    - Strips leading/trailing whitespace from each line.
    - Collapses multiple blank lines into a single blank line.
    """
    # Replace the ideographic (full-width) space with a standard space.
    text = text.replace('\u3000', ' ')
    
    lines = text.split('\n')
    cleaned_lines = []
    # Use a flag to track consecutive empty lines
    was_last_line_empty = False
    
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            # This is an empty line
            if not was_last_line_empty:
                cleaned_lines.append('') # Add one empty line
                was_last_line_empty = True
        else:
            # This line has content
            cleaned_lines.append(stripped_line)
            was_last_line_empty = False
            
    return '\n'.join(cleaned_lines).strip()

def convert_with_pandoc(source_path: Path) -> str:
    """
    Converts DOCX, DOC, and EPUB files to clean Markdown using pypandoc.
    This function is specifically configured to strip out unwanted elements.
    """
    file_type = source_path.suffix.upper()[1:]
    print(f"      -> Using pypandoc for {file_type} conversion...")
    
    extra_args = [
        '--no-highlight',
        '--strip-comments',
        '--extract-media=""',
        '--wrap=none'
    ]
    
    markdown_string = pypandoc.convert_file(
        str(source_path),
        to='gfm',
        extra_args=extra_args
    )
    
    cleaned_markdown = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_string)
    print(f"      -> {file_type} converted and cleaned.")
    return cleaned_markdown

def convert_pdf_with_pymupdf(source_path: Path) -> str:
    """
    Extracts raw text from a PDF file using PyMuPDF and cleans it.
    This method focuses on text content only.
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

        print(f"      -> PDF text extracted from {num_pages} pages and cleaned.")
        return cleaned_text
    except Exception as e:
        print(f"      -> Error during PDF processing: {e}")
        raise

# --- 2. FILE SPLITTING LOGIC ---

def split_markdown_by_lesson(md_file_path: Path):
    """
    Reads a markdown file and splits it into multiple files based on a lesson pattern.
    It now correctly captures content before the first lesson marker.
    """
    print(f"   -> Checking for lesson markers in '{md_file_path.name}'...")
    lesson_pattern = re.compile(r"^(第.*课)$")
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lessons = []
        current_lesson_content = []
        # Start by collecting content for a potential "introduction" part.
        current_lesson_title = "_part00_introduction" 

        for line in lines:
            match = lesson_pattern.match(line.strip())
            if match:
                # A new lesson marker is found.
                # First, save the content of the PREVIOUS part (intro or prior lesson).
                if current_lesson_content:
                    lessons.append((current_lesson_title, "".join(current_lesson_content)))
                
                # Now, start the new lesson.
                # Sanitize the title for use in a filename.
                safe_title = re.sub(r'[\\/*?:"<>|]', "", match.group(1).strip())
                current_lesson_title = safe_title
                current_lesson_content = [line] # Start new content with the title line
            else:
                # Not a marker, so append the line to the current part.
                current_lesson_content.append(line)
        
        # After the loop, save the very last lesson found.
        if current_lesson_content:
            lessons.append((current_lesson_title, "".join(current_lesson_content)))

        # If only one "lesson" was found (either just intro or no markers), don't split.
        if len(lessons) <= 1:
            print("   -> No lesson markers found or only one section. No splitting performed.")
            return

        print(f"   -> Found {len(lessons)} parts. Saving split files...")
        base_name = md_file_path.stem
        output_dir = md_file_path.parent

        for i, (title, content) in enumerate(lessons):
            # Use a numeric prefix to keep files in order.
            new_filename = f"{base_name}_{i:02d}_{title}.md"
            output_path = output_dir / new_filename
            
            # Clean whitespace again on the final chunk before saving.
            final_content = _clean_and_normalize_whitespace(content)
            output_path.write_text(final_content, encoding='utf-8')
            print(f"      - Saved lesson split: '{output_path.name}'")
            
    except Exception as e:
        print(f"   -> ERROR during splitting of '{md_file_path.name}': {e}")
        traceback.print_exc()

# --- 3. MAIN DISPATCHER AND BATCH PROCESSING ---

def convert_file(source_path: Path, output_path: Path) -> bool:
    """
    Dispatcher function that routes a file to the correct conversion utility,
    then applies whitespace cleaning before saving.
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

    # Apply the final whitespace cleaning and normalization step.
    final_content = _clean_and_normalize_whitespace(markdown_content)
    
    # Write the initial, unsplit, and cleaned markdown file.
    output_path.write_text(final_content, encoding='utf-8')
    return True

def batch_process_directory(source_dir: Path, output_dir: Path):
    """
    Main batch processing loop. It finds all supported files in a source
    directory, converts them, and then attempts to split them.
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
        return

    total_files = len(source_files)
    print(f"✅ Found {total_files} file(s) to process.")
    
    success_count, failure_count = 0, 0

    for i, source_path in enumerate(source_files, 1):
        print(f"\n[{i}/{total_files}] Processing: '{source_path}'")

        try:
            relative_path = source_path.relative_to(source_dir)
            output_md_path = output_dir / relative_path.with_suffix(".md")
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   -> Converting to: '{output_md_path}'")
            conversion_success = convert_file(source_path, output_md_path)

            if conversion_success:
                split_markdown_by_lesson(output_md_path)
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
    print("Starting document conversion and splitting process...")
    batch_process_directory(SOURCE_DIR, OUTPUT_DIR)
    print("\nAll processing complete.")

if __name__ == "__main__":
    main()