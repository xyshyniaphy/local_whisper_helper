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

# --- 1. SPECIALIZED CONVERSION FUNCTIONS ---

def convert_with_pandoc(source_path: Path) -> str:
    """
    Converts DOCX, DOC, and EPUB files to clean Markdown using pypandoc.
    This function is specifically configured to strip out unwanted elements.
    """
    file_type = source_path.suffix.upper()[1:]
    print(f"      -> Using pypandoc for {file_type} conversion...")
    
    # Pandoc arguments to enforce cleaning rules:
    # --no-highlight: Prevents syntax highlighting and associated CSS/styles.
    # --strip-comments: Removes any comments from the document.
    # --extract-media="": An empty path prevents any images from being extracted or linked.
    # --wrap=none: Prevents pandoc from adding hard line breaks for better flow.
    extra_args = [
        '--no-highlight',
        '--strip-comments',
        '--extract-media=""',
        '--wrap=none'
    ]
    
    # Convert the file to GitHub Flavored Markdown (gfm) for good structure.
    # The 'extra_args' will handle the cleaning.
    markdown_string = pypandoc.convert_file(
        str(source_path),
        to='gfm',
        extra_args=extra_args
    )
    
    # Post-processing with regex to remove any remaining hyperlinks.
    # This pattern finds [link text](url) and replaces it with just "link text".
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
            # FIX: Get the page count while the document is open and store it.
            num_pages = len(doc)
            for page_num, page in enumerate(doc, 1):
                full_text += page.get_text() + "\n"
        
        # Post-processing with regex to remove URLs and email addresses from the raw text.
        # Pattern for standard URLs (http, https, ftp)
        cleaned_text = re.sub(r'(https?|ftp)://[^\s/$.?#].[^\s]*', '', full_text, flags=re.IGNORECASE)
        # Pattern for www. links
        cleaned_text = re.sub(r'www\.[^\s/$.?#].[^\s]*', '', cleaned_text, flags=re.IGNORECASE)
        # Pattern for email addresses
        cleaned_text = re.sub(r'\S+@\S+\.\S+', '', cleaned_text)

        # FIX: Use the stored 'num_pages' variable instead of accessing the closed 'doc' object.
        print(f"      -> PDF text extracted from {num_pages} pages and cleaned.")
        return cleaned_text
    except Exception as e:
        print(f"      -> Error during PDF processing: {e}")
        raise # Re-raise the exception to be caught by the main loop

# --- 2. FILE SPLITTING LOGIC ---

def split_markdown_by_lesson(md_file_path: Path):
    """
    Reads a markdown file and splits it into multiple files based on a lesson pattern.
    The pattern is a line that starts with "第" and ends with "课".
    """
    print(f"   -> Checking for lesson markers in '{md_file_path.name}'...")
    
    # This regex pattern matches a line that consists ONLY of the lesson title.
    # ^ asserts position at start of the string, $ asserts position at the end.
    lesson_pattern = re.compile(r"^(第.*课)$")
    
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        lessons = []
        current_lesson_content = []
        current_lesson_title = None

        for line in lines:
            # We strip the line to check for an exact match (e.g., "第1课")
            match = lesson_pattern.match(line.strip())
            
            if match:
                # A new lesson marker is found.
                # First, save the content of the PREVIOUS lesson if it exists.
                if current_lesson_title:
                    lessons.append((current_lesson_title, "".join(current_lesson_content)))
                
                # Now, start the new lesson.
                current_lesson_title = match.group(1).strip()
                current_lesson_content = [line] # Start the new content with the title line
            elif current_lesson_title:
                # If we are currently inside a lesson, append the line to its content.
                current_lesson_content.append(line)
        
        # After the loop, save the very last lesson found.
        if current_lesson_title:
            lessons.append((current_lesson_title, "".join(current_lesson_content)))

        if not lessons:
            print("   -> No lesson markers found. No splitting performed.")
            return

        # If lessons were found, write them to new files.
        print(f"   -> Found {len(lessons)} lessons. Saving split files...")
        base_name = md_file_path.stem
        output_dir = md_file_path.parent

        for title, content in lessons:
            # Create a safe filename from the lesson title
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
            new_filename = f"{base_name}_{safe_title}.md"
            output_path = output_dir / new_filename
            
            output_path.write_text(content.strip(), encoding='utf-8')
            print(f"      - Saved lesson split: '{output_path.name}'")
            
    except Exception as e:
        print(f"   -> ERROR during splitting of '{md_file_path.name}': {e}")
        traceback.print_exc()

# --- 3. MAIN DISPATCHER AND BATCH PROCESSING ---

def convert_file(source_path: Path, output_path: Path) -> bool:
    """
    Dispatcher function that routes a file to the correct conversion utility
    based on its extension.
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

    # Write the initial, unsplit markdown file
    output_path.write_text(markdown_content, encoding='utf-8')
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

    # Find all supported files recursively
    supported_patterns = ["*.doc", "*.docx", "*.pdf", "*.epub"]
    source_files = []
    for pattern in supported_patterns:
        source_files.extend(list(source_dir.rglob(pattern)))

    if not source_files:
        print("No supported files found to process.")
        return

    total_files = len(source_files)
    print(f"✅ Found {total_files} file(s) to process.")
    
    success_count, failure_count = 0, 0

    for i, source_path in enumerate(source_files, 1):
        print(f"\n[{i}/{total_files}] Processing: '{source_path}'")

        try:
            # **REVISED FOLDER LOGIC**
            # Get the file's path relative to the source directory.
            relative_path = source_path.relative_to(source_dir)
            # Construct the full output path, preserving the subdirectory structure.
            output_md_path = output_dir / relative_path.with_suffix(".md")
            
            # Ensure the parent directory for the output file exists.
            output_md_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"   -> Converting to: '{output_md_path}'")
            conversion_success = convert_file(source_path, output_md_path)

            if conversion_success:
                # If conversion was successful, proceed to the splitting step
                split_markdown_by_lesson(output_md_path)
                print(f"   ✅ Successfully processed '{source_path.name}'.")
                success_count += 1
            else:
                # If conversion failed or was skipped, count as a failure
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
    
    # Process the single source directory
    batch_process_directory(SOURCE_DIR, OUTPUT_DIR)
    
    print("\nAll processing complete.")

if __name__ == "__main__":
    main()